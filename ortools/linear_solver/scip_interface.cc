// Copyright 2010-2018 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(USE_SCIP)

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "ortools/base/canonical_errors.h"
#include "ortools/base/commandlineflags.h"
#include "ortools/base/hash.h"
#include "ortools/base/integral_types.h"
#include "ortools/base/logging.h"
#include "ortools/base/status.h"
#include "ortools/base/status_macros.h"
#include "ortools/base/timer.h"
#include "ortools/base/dynamic_library.h"
#include "ortools/linear_solver/linear_solver.h"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"

DEFINE_bool(scip_feasibility_emphasis, false,
            "When true, emphasize search towards feasibility. This may or "
            "may not result in speedups in some problems.");

namespace operations_research {

class SCIPInterface : public MPSolverInterface {
 public:
  explicit SCIPInterface(MPSolver* solver);
  ~SCIPInterface() override;

  void SetOptimizationDirection(bool maximize) override;
  MPSolver::ResultStatus Solve(const MPSolverParameters& param) override;
  void Reset() override;

  void SetVariableBounds(int var_index, double lb, double ub) override;
  void SetVariableInteger(int var_index, bool integer) override;
  void SetConstraintBounds(int row_index, double lb, double ub) override;

  void AddRowConstraint(MPConstraint* ct) override;
  void AddVariable(MPVariable* var) override;
  void SetCoefficient(MPConstraint* constraint, const MPVariable* variable,
                      double new_value, double old_value) override;
  void ClearConstraint(MPConstraint* constraint) override;
  void SetObjectiveCoefficient(const MPVariable* variable,
                               double coefficient) override;
  void SetObjectiveOffset(double value) override;
  void ClearObjective() override;

  int64 iterations() const override;
  int64 nodes() const override;
  double best_objective_bound() const override;
  MPSolver::BasisStatus row_status(int constraint_index) const override {
    LOG(DFATAL) << "Basis status only available for continuous problems";
    return MPSolver::FREE;
  }
  MPSolver::BasisStatus column_status(int variable_index) const override {
    LOG(DFATAL) << "Basis status only available for continuous problems";
    return MPSolver::FREE;
  }

  bool IsContinuous() const override { return false; }
  bool IsLP() const override { return false; }
  bool IsMIP() const override { return true; }

  void ExtractNewVariables() override;
  void ExtractNewConstraints() override;
  void ExtractObjective() override;

  std::string SolverVersion() const override {
    return absl::StrFormat("SCIP %d.%d.%d [LP solver: %s]", SCIPmajorVersion(),
                           SCIPminorVersion(), SCIPtechVersion(),
                           SCIPlpiGetSolverName());
  }

  bool InterruptSolve() override {
    if (scip_ == nullptr) return true;  // NOTE(user): Is this weird?
    return SCIPinterruptSolve(scip_) == SCIP_OKAY;
  }

  void* underlying_solver() override { return reinterpret_cast<void*>(scip_); }

 private:
  void SetParameters(const MPSolverParameters& param) override;
  void SetRelativeMipGap(double value) override;
  void SetPrimalTolerance(double value) override;
  void SetDualTolerance(double value) override;
  void SetPresolveMode(int value) override;
  void SetScalingMode(int value) override;
  void SetLpAlgorithm(int value) override;

  // SCIP parameters allow to lower and upper bound the number of threads used
  // (via "parallel/minnthreads" and "parallel/maxnthread", respectively). Here,
  // we interpret "num_threads" to mean "parallel/maxnthreads", as this is what
  // most clients probably want to do. To change "parallel/minnthreads" use
  // SetSolverSpecificParametersAsString(). However, one must change
  // "parallel/maxnthread" with SetNumThreads() because only this will inform
  // the interface to run SCIPsolveConcurrent() instead of SCIPsolve() which is
  // necessery to enable multi-threading.
  util::Status SetNumThreads(int num_threads) override;

  bool ReadParameterFile(const std::string& filename) override;
  std::string ValidFileExtensionForParameterFile() const override;

  void SetUnsupportedIntegerParam(
      MPSolverParameters::IntegerParam param) override;
  void SetIntegerParamToUnsupportedValue(MPSolverParameters::IntegerParam param,
                                         int value) override;

  util::Status CreateSCIP();
  void DeleteSCIP();

  // SCIP has many internal checks (many of which are numerical) that can fail
  // during various phases: upon startup, when loading the model, when solving,
  // etc. Often, the user is meant to stop at the first error, but since most
  // of the linear solver interface API doesn't support "error reporting", we
  // store a potential error status here.
  // If this status isn't OK, then most operations will silently be cancelled.
  util::Status status_;

  SCIP* scip_;
  SCIP_VAR* objective_offset_variable_;
  std::vector<SCIP_VAR*> scip_variables_;
  std::vector<SCIP_CONS*> scip_constraints_;
  DynamicLibrary* lib_;

  std::function<SCIP_RETCODE(SCIP*, SCIP_CONS*, SCIP_VAR*, SCIP_Real)>
      SCIPaddCoefLinear;
  std::function<SCIP_RETCODE(SCIP*, SCIP_CONS*)> SCIPaddCons;
  std::function<SCIP_RETCODE(SCIP*, SCIP_SOL**, SCIP_Bool*)> SCIPaddSolFree;
  std::function<SCIP_RETCODE(SCIP*, SCIP_VAR*)> SCIPaddVar;
  std::function<SCIP_RETCODE(SCIP*, SCIP_SOL*, SCIP_Bool, SCIP_Bool,
                             SCIP_Bool, SCIP_Bool, SCIP_Bool, SCIP_Bool*)>
      SCIPcheckSol;
  std::function<SCIP_RETCODE(SCIP*, SCIP_CONS*, SCIP_Real)> SCIPchgLhsLinear;
  std::function<SCIP_RETCODE(SCIP*, SCIP_CONS*, SCIP_Real)> SCIPchgRhsLinear;
  std::function<SCIP_RETCODE(SCIP*, SCIP_VAR*, SCIP_Real)> SCIPchgVarLb;
  std::function<SCIP_RETCODE(SCIP*, SCIP_VAR*, SCIP_Real)> SCIPchgVarObj;
  std::function<SCIP_RETCODE(SCIP*, SCIP_VAR*, SCIP_VARTYPE, SCIP_Bool*)>
      SCIPchgVarType;
  std::function<SCIP_RETCODE(SCIP*, SCIP_VAR*, SCIP_Real)> SCIPchgVarUb;
  std::function<SCIP_RETCODE(SCIP**)> SCIPcreate;
  std::function<SCIP_RETCODE(
      SCIP*, SCIP_CONS**, const char*, int, SCIP_VAR**, SCIP_Real*,
      SCIP_Real, SCIP_Real, SCIP_Bool, SCIP_Bool, SCIP_Bool, SCIP_Bool,
      SCIP_Bool, SCIP_Bool, SCIP_Bool, SCIP_Bool, SCIP_Bool, SCIP_Bool)>
      SCIPcreateConsLinear;
  std::function<SCIP_RETCODE(SCIP*, SCIP_SOL**, SCIP_HEUR*)>
      SCIPcreatePartialSol;
  std::function<SCIP_RETCODE(SCIP*, const char*, SCIP_DECL_PROBDELORIG(*),
                             SCIP_DECL_PROBTRANS(*), SCIP_DECL_PROBDELTRANS(*),
                             SCIP_DECL_PROBINITSOL(*), SCIP_DECL_PROBEXITSOL(*),
                             SCIP_DECL_PROBCOPY(*), SCIP_PROBDATA*)>
      SCIPcreateProb;
  std::function<SCIP_RETCODE(SCIP*, SCIP_SOL**, SCIP_HEUR*)> SCIPcreateSol;
  std::function<SCIP_RETCODE(SCIP*, SCIP_VAR**, const char*, SCIP_Real,
                             SCIP_Real, SCIP_Real, SCIP_VARTYPE, SCIP_Bool,
                             SCIP_Bool, SCIP_DECL_VARDELORIG(*),
                             SCIP_DECL_VARTRANS(*), SCIP_DECL_VARDELTRANS(*),
                             SCIP_DECL_VARCOPY(*), SCIP_VARDATA*)>
      SCIPcreateVar;
  std::function<SCIP_RETCODE(SCIP**)> SCIPfree;
  std::function<SCIP_RETCODE(SCIP*)> SCIPfreeTransform;
  std::function<SCIP_SOL*(SCIP*)> SCIPgetBestSol;
  std::function<SCIP_Real(SCIP*)> SCIPgetDualbound;
  std::function<SCIP_Longint(SCIP*)> SCIPgetNLPIterations;
  std::function<SCIP_Longint(SCIP*)> SCIPgetNTotalNodes;
  std::function<SCIP_Real(SCIP*, SCIP_SOL*)> SCIPgetSolOrigObj;
  std::function<SCIP_Real(SCIP*, SCIP_SOL*, SCIP_VAR*)> SCIPgetSolVal;
  std::function<SCIP_STATUS(SCIP*)> SCIPgetStatus;
  std::function<SCIP_RETCODE(SCIP*)> SCIPincludeDefaultPlugins;
  std::function<SCIP_RETCODE(SCIP*)> SCIPinterruptSolve;
  std::function<SCIP_Bool(SCIP*)> SCIPisTransformed;
  std::function<const char*(void)> SCIPlpiGetSolverName;
  std::function<int(void)> SCIPmajorVersion;
  std::function<int(void)> SCIPminorVersion;
  std::function<SCIP_RETCODE(SCIP*, const char*)> SCIPreadParams;
  std::function<SCIP_RETCODE(SCIP*, const char*)> SCIPreadSol;
  std::function<SCIP_RETCODE(SCIP*, SCIP_CONS**)> SCIPreleaseCons;
  std::function<SCIP_RETCODE(SCIP*, SCIP_VAR**)> SCIPreleaseVar;
  std::function<SCIP_RETCODE(SCIP*, const char*)> SCIPresetParam;
  std::function<SCIP_RETCODE(SCIP*)> SCIPresetParams;
  std::function<SCIP_RETCODE(SCIP*, const char*, char)> SCIPsetCharParam;
  std::function<SCIP_RETCODE(SCIP*, SCIP_PARAMEMPHASIS, SCIP_Bool)>
      SCIPsetEmphasis;
  std::function<SCIP_RETCODE(SCIP*, const char*, int)> SCIPsetIntParam;
  std::function<void(SCIP*, SCIP_Bool)> SCIPsetMessagehdlrQuiet;
  std::function<SCIP_RETCODE(SCIP*, SCIP_OBJSENSE)> SCIPsetObjsense;
  std::function<SCIP_RETCODE(SCIP*, const char*, SCIP_Real)> SCIPsetRealParam;
  std::function<SCIP_RETCODE(SCIP*, SCIP_SOL*, SCIP_VAR*, SCIP_Real)>
      SCIPsetSolVal;
  std::function<SCIP_RETCODE(SCIP*)> SCIPsolve;
  std::function<int(void)> SCIPtechVersion;
  std::function<SCIP_RETCODE(SCIP*, SCIP_SOL**, SCIP_Bool, SCIP_Bool,
                             SCIP_Bool, SCIP_Bool, SCIP_Bool, SCIP_Bool*)>
      SCIPtrySolFree;
};

// Our own version of SCIP_CALL to do error management.
// NOTE(user): There are so many SCIP error codes, in so many different
// situations.. We don't try to match them perfectly to google3 error codes.
// Instead, we use the most likely/generic code "invalid argument" and surface
// the internal SCIP error code to the user.
#define TO_STATUS(x) ScipReturnCodeToUtilStatus(x, __FILE__, __LINE__, #x)
util::Status ScipReturnCodeToUtilStatus(SCIP_Retcode retcode,
                                        const char* source_file,
                                        int source_line,
                                        const char* scip_statement) {
  if (retcode == SCIP_OKAY) return util::OkStatus();
  return util::InvalidArgumentError(
      absl::StrFormat("SCIP error code %d (file '%s', line %d) on '%s'",
                      retcode, source_file, source_line, scip_statement));
}

SCIPInterface::SCIPInterface(MPSolver* solver)
    : MPSolverInterface(solver), scip_(nullptr) {
  try {
    std::string library_name =
#if defined(_MSC_VER)
    "scip.dll";
#elif defined(__GNUC__)
    "libscip.so";
#endif
    if (!solver_->library_name_.empty()) {
      library_name = solver_->library_name_;
    }
    
    lib_ = new DynamicLibrary(library_name);
    lib_->GetFunction(&SCIPaddCoefLinear, NAMEOF(SCIPaddCoefLinear));
    lib_->GetFunction(&SCIPaddCons, NAMEOF(SCIPaddCons));
    lib_->GetFunction(&SCIPaddSolFree, NAMEOF(SCIPaddSolFree));
    lib_->GetFunction(&SCIPaddVar, NAMEOF(SCIPaddVar));
    lib_->GetFunction(&SCIPcheckSol, NAMEOF(SCIPcheckSol));
    lib_->GetFunction(&SCIPchgLhsLinear, NAMEOF(SCIPchgLhsLinear));
    lib_->GetFunction(&SCIPchgRhsLinear, NAMEOF(SCIPchgRhsLinear));
    lib_->GetFunction(&SCIPchgVarLb, NAMEOF(SCIPchgVarLb));
    lib_->GetFunction(&SCIPchgVarObj, NAMEOF(SCIPchgVarObj));
    lib_->GetFunction(&SCIPchgVarType, NAMEOF(SCIPchgVarType));
    lib_->GetFunction(&SCIPchgVarUb, NAMEOF(SCIPchgVarUb));
    lib_->GetFunction(&SCIPcreate, NAMEOF(SCIPcreate));
    lib_->GetFunction(&SCIPcreateConsLinear, NAMEOF(SCIPcreateConsLinear));
    lib_->GetFunction(&SCIPcreatePartialSol, NAMEOF(SCIPcreatePartialSol));
    lib_->GetFunction(&SCIPcreateProb, NAMEOF(SCIPcreateProb));
    lib_->GetFunction(&SCIPcreateSol, NAMEOF(SCIPcreateSol));
    lib_->GetFunction(&SCIPcreateVar, NAMEOF(SCIPcreateVar));
    lib_->GetFunction(&SCIPfree, NAMEOF(SCIPfree));
    lib_->GetFunction(&SCIPfreeTransform, NAMEOF(SCIPfreeTransform));
    lib_->GetFunction(&SCIPgetBestSol, NAMEOF(SCIPgetBestSol));
    lib_->GetFunction(&SCIPgetDualbound, NAMEOF(SCIPgetDualbound));
    lib_->GetFunction(&SCIPgetNLPIterations, NAMEOF(SCIPgetNLPIterations));
    lib_->GetFunction(&SCIPgetNTotalNodes, NAMEOF(SCIPgetNTotalNodes));
    lib_->GetFunction(&SCIPgetSolOrigObj, NAMEOF(SCIPgetSolOrigObj));
    lib_->GetFunction(&SCIPgetSolVal, NAMEOF(SCIPgetSolVal));
    lib_->GetFunction(&SCIPgetStatus, NAMEOF(SCIPgetStatus));
    lib_->GetFunction(&SCIPincludeDefaultPlugins,
                      NAMEOF(SCIPincludeDefaultPlugins));
    lib_->GetFunction(&SCIPinterruptSolve, NAMEOF(SCIPinterruptSolve));
    lib_->GetFunction(&SCIPisTransformed, NAMEOF(SCIPisTransformed));
    lib_->GetFunction(&SCIPlpiGetSolverName, NAMEOF(SCIPlpiGetSolverName));
    lib_->GetFunction(&SCIPmajorVersion, NAMEOF(SCIPmajorVersion));
    lib_->GetFunction(&SCIPminorVersion, NAMEOF(SCIPminorVersion));
    lib_->GetFunction(&SCIPreadParams, NAMEOF(SCIPreadParams));
    lib_->GetFunction(&SCIPreadSol, NAMEOF(SCIPreadSol));
    lib_->GetFunction(&SCIPreleaseCons, NAMEOF(SCIPreleaseCons));
    lib_->GetFunction(&SCIPreleaseVar, NAMEOF(SCIPreleaseVar));
    lib_->GetFunction(&SCIPresetParam, NAMEOF(SCIPresetParam));
    lib_->GetFunction(&SCIPresetParams, NAMEOF(SCIPresetParams));
    lib_->GetFunction(&SCIPsetCharParam, NAMEOF(SCIPsetCharParam));
    lib_->GetFunction(&SCIPsetEmphasis, NAMEOF(SCIPsetEmphasis));
    lib_->GetFunction(&SCIPsetIntParam, NAMEOF(SCIPsetIntParam));
    lib_->GetFunction(&SCIPsetMessagehdlrQuiet,
                      NAMEOF(SCIPsetMessagehdlrQuiet));
    lib_->GetFunction(&SCIPsetObjsense, NAMEOF(SCIPsetObjsense));
    lib_->GetFunction(&SCIPsetRealParam, NAMEOF(SCIPsetRealParam));
    lib_->GetFunction(&SCIPsetSolVal, NAMEOF(SCIPsetSolVal));
    lib_->GetFunction(&SCIPsolve, NAMEOF(SCIPsolve));
    lib_->GetFunction(&SCIPtechVersion, NAMEOF(SCIPtechVersion));
    lib_->GetFunction(&SCIPtrySolFree, NAMEOF(SCIPtrySolFree));
  } catch (const std::runtime_error& e) {
    LOG(DFATAL) << e.what();
    delete lib_;
    throw;
  }

  status_ = CreateSCIP();
}

SCIPInterface::~SCIPInterface() {
  DeleteSCIP();
  delete lib_;
}

void SCIPInterface::Reset() {
  DeleteSCIP();
  status_ = CreateSCIP();
  ResetExtractionInformation();
}

util::Status SCIPInterface::CreateSCIP() {
  RETURN_IF_ERROR(TO_STATUS(SCIPcreate(&scip_)));
  RETURN_IF_ERROR(TO_STATUS(SCIPincludeDefaultPlugins(scip_)));
  // Set the emphasis to enum SCIP_PARAMEMPHASIS_FEASIBILITY. Do not print
  // the new parameter (quiet = true).
  if (FLAGS_scip_feasibility_emphasis) {
    RETURN_IF_ERROR(
        TO_STATUS(SCIPsetEmphasis(scip_, SCIP_PARAMEMPHASIS_FEASIBILITY,
                                  /*quiet=*/true)));
  }
  // Default clock type. We use wall clock time because getting CPU user seconds
  // involves calling times() which is very expensive.
  RETURN_IF_ERROR(TO_STATUS(
      SCIPsetIntParam(scip_, "timing/clocktype", SCIP_CLOCKTYPE_WALL)));
  RETURN_IF_ERROR(
      TO_STATUS(SCIPcreateProb(scip_, solver_->name_.c_str(), nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr)));
  RETURN_IF_ERROR(TO_STATUS(SCIPsetObjsense(
      scip_, maximize_ ? SCIP_OBJSENSE_MAXIMIZE : SCIP_OBJSENSE_MINIMIZE)));
  // SCIPaddObjoffset cannot be used at the problem building stage. So we handle
  // the objective offset by creating a dummy variable.
  objective_offset_variable_ = nullptr;
  // The true objective coefficient will be set in ExtractObjective.
  double dummy_obj_coef = 0.0;
  RETURN_IF_ERROR(TO_STATUS(
      SCIPcreateVar(scip_, &objective_offset_variable_, "dummy", 1.0, 1.0,
                    dummy_obj_coef, SCIP_VARTYPE_CONTINUOUS, true, false,
                    nullptr, nullptr, nullptr, nullptr, nullptr)));
  RETURN_IF_ERROR(TO_STATUS(SCIPaddVar(scip_, objective_offset_variable_)));
  return util::OkStatus();
}

void SCIPInterface::DeleteSCIP() {
  // NOTE(user): DeleteSCIP() shouldn't "give up" mid-stage if it fails, since
  // it might be the user's chance to reset the solver to start fresh without
  // errors. The current code isn't perfect, since some CHECKs() remain, but
  // hopefully they'll never be triggered in practice.
  CHECK(scip_ != nullptr);
  if (objective_offset_variable_ != nullptr) {
    CHECK_EQ(SCIPreleaseVar(scip_, &objective_offset_variable_), SCIP_OKAY);
  }
  for (int i = 0; i < scip_variables_.size(); ++i) {
    CHECK_EQ(SCIPreleaseVar(scip_, &scip_variables_[i]), SCIP_OKAY);
  }
  scip_variables_.clear();
  for (int j = 0; j < scip_constraints_.size(); ++j) {
    CHECK_EQ(SCIPreleaseCons(scip_, &scip_constraints_[j]), SCIP_OKAY);
  }
  scip_constraints_.clear();
  CHECK_EQ(SCIPfree(&scip_), SCIP_OKAY);
  scip_ = nullptr;
}

#define RETURN_IF_ALREADY_IN_ERROR_STATE                                 \
  do {                                                                   \
    if (!status_.ok()) {                                                 \
      VLOG_EVERY_N_SEC(1, 10) << "Early abort: SCIP is in error state."; \
      return;                                                            \
    }                                                                    \
  } while (false)

#define RETURN_IF_SCIP_ERROR(x) \
  do {                          \
    status_ = TO_STATUS(x);     \
    if (!status_.ok()) return;  \
  } while (false)

// Not cached.
void SCIPInterface::SetOptimizationDirection(bool maximize) {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  InvalidateSolutionSynchronization();
  RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
  RETURN_IF_SCIP_ERROR(SCIPsetObjsense(
      scip_, maximize ? SCIP_OBJSENSE_MAXIMIZE : SCIP_OBJSENSE_MINIMIZE));
}

void SCIPInterface::SetVariableBounds(int var_index, double lb, double ub) {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  InvalidateSolutionSynchronization();
  if (variable_is_extracted(var_index)) {
    // Not cached if the variable has been extracted.
    DCHECK_LT(var_index, last_variable_index_);
    RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
    RETURN_IF_SCIP_ERROR(SCIPchgVarLb(scip_, scip_variables_[var_index], lb));
    RETURN_IF_SCIP_ERROR(SCIPchgVarUb(scip_, scip_variables_[var_index], ub));
  } else {
    sync_status_ = MUST_RELOAD;
  }
}

void SCIPInterface::SetVariableInteger(int var_index, bool integer) {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  InvalidateSolutionSynchronization();
  if (variable_is_extracted(var_index)) {
    // Not cached if the variable has been extracted.
    RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
#if (SCIP_VERSION >= 210)
    SCIP_Bool infeasible = false;
    RETURN_IF_SCIP_ERROR(SCIPchgVarType(
        scip_, scip_variables_[var_index],
        integer ? SCIP_VARTYPE_INTEGER : SCIP_VARTYPE_CONTINUOUS, &infeasible));
#else
    RETURN_IF_SCIP_ERROR(SCIPchgVarType(
        scip_, scip_variables_[var_index],
        integer ? SCIP_VARTYPE_INTEGER : SCIP_VARTYPE_CONTINUOUS));
#endif  // SCIP_VERSION >= 210
  } else {
    sync_status_ = MUST_RELOAD;
  }
}

void SCIPInterface::SetConstraintBounds(int index, double lb, double ub) {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  InvalidateSolutionSynchronization();
  if (constraint_is_extracted(index)) {
    // Not cached if the row has been extracted.
    DCHECK_LT(index, last_constraint_index_);
    RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
    RETURN_IF_SCIP_ERROR(SCIPchgLhsLinear(scip_, scip_constraints_[index], lb));
    RETURN_IF_SCIP_ERROR(SCIPchgRhsLinear(scip_, scip_constraints_[index], ub));
  } else {
    sync_status_ = MUST_RELOAD;
  }
}

void SCIPInterface::SetCoefficient(MPConstraint* constraint,
                                   const MPVariable* variable, double new_value,
                                   double old_value) {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  InvalidateSolutionSynchronization();
  if (variable_is_extracted(variable->index()) &&
      constraint_is_extracted(constraint->index())) {
    // The modification of the coefficient for an extracted row and
    // variable is not cached.
    DCHECK_LT(constraint->index(), last_constraint_index_);
    DCHECK_LT(variable->index(), last_variable_index_);
    // SCIP does not allow to set a coefficient directly, so we add the
    // difference between the new and the old value instead.
    RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
    RETURN_IF_SCIP_ERROR(SCIPaddCoefLinear(
        scip_, scip_constraints_[constraint->index()],
        scip_variables_[variable->index()], new_value - old_value));
  } else {
    // The modification of an unextracted row or variable is cached
    // and handled in ExtractModel.
    sync_status_ = MUST_RELOAD;
  }
}

// Not cached
void SCIPInterface::ClearConstraint(MPConstraint* constraint) {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  InvalidateSolutionSynchronization();
  const int constraint_index = constraint->index();
  // Constraint may not have been extracted yet.
  if (!constraint_is_extracted(constraint_index)) return;
  for (const auto& entry : constraint->coefficients_) {
    const int var_index = entry.first->index();
    const double old_coef_value = entry.second;
    DCHECK(variable_is_extracted(var_index));
    RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
    // Set coefficient to zero by substracting the old coefficient value.
    RETURN_IF_SCIP_ERROR(
        SCIPaddCoefLinear(scip_, scip_constraints_[constraint_index],
                          scip_variables_[var_index], -old_coef_value));
  }
}

// Cached
void SCIPInterface::SetObjectiveCoefficient(const MPVariable* variable,
                                            double coefficient) {
  sync_status_ = MUST_RELOAD;
}

// Cached
void SCIPInterface::SetObjectiveOffset(double value) {
  sync_status_ = MUST_RELOAD;
}

// Clear objective of all its terms.
void SCIPInterface::ClearObjective() {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  InvalidateSolutionSynchronization();
  RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
  // Clear linear terms
  for (const auto& entry : solver_->objective_->coefficients_) {
    const int var_index = entry.first->index();
    // Variable may have not been extracted yet.
    if (!variable_is_extracted(var_index)) {
      DCHECK_NE(MODEL_SYNCHRONIZED, sync_status_);
    } else {
      RETURN_IF_SCIP_ERROR(
          SCIPchgVarObj(scip_, scip_variables_[var_index], 0.0));
    }
  }
  // Constant term: change objective offset variable.
  RETURN_IF_SCIP_ERROR(SCIPchgVarObj(scip_, objective_offset_variable_, 0.0));
}

void SCIPInterface::AddRowConstraint(MPConstraint* ct) {
  sync_status_ = MUST_RELOAD;
}

void SCIPInterface::AddVariable(MPVariable* var) { sync_status_ = MUST_RELOAD; }

void SCIPInterface::ExtractNewVariables() {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  int total_num_vars = solver_->variables_.size();
  if (total_num_vars > last_variable_index_) {
    RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
    // Define new variables
    for (int j = last_variable_index_; j < total_num_vars; ++j) {
      MPVariable* const var = solver_->variables_[j];
      DCHECK(!variable_is_extracted(j));
      set_variable_as_extracted(j, true);
      SCIP_VAR* scip_var = nullptr;
      // The true objective coefficient will be set later in ExtractObjective.
      double tmp_obj_coef = 0.0;
      RETURN_IF_SCIP_ERROR(SCIPcreateVar(
          scip_, &scip_var, var->name().c_str(), var->lb(), var->ub(),
          tmp_obj_coef,
          var->integer() ? SCIP_VARTYPE_INTEGER : SCIP_VARTYPE_CONTINUOUS, true,
          false, nullptr, nullptr, nullptr, nullptr, nullptr));
      RETURN_IF_SCIP_ERROR(SCIPaddVar(scip_, scip_var));
      scip_variables_.push_back(scip_var);
    }
    // Add new variables to existing constraints.
    for (int i = 0; i < last_constraint_index_; i++) {
      MPConstraint* const ct = solver_->constraints_[i];
      for (const auto& entry : ct->coefficients_) {
        const int var_index = entry.first->index();
        DCHECK(variable_is_extracted(var_index));
        if (var_index >= last_variable_index_) {
          // The variable is new, so we know the previous coefficient
          // value was 0 and we can directly add the coefficient.
          RETURN_IF_SCIP_ERROR(SCIPaddCoefLinear(scip_, scip_constraints_[i],
                                                 scip_variables_[var_index],
                                                 entry.second));
        }
      }
    }
  }
}

void SCIPInterface::ExtractNewConstraints() {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  int total_num_rows = solver_->constraints_.size();
  if (last_constraint_index_ < total_num_rows) {
    RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
    // Find the length of the longest row.
    int max_row_length = 0;
    for (int i = last_constraint_index_; i < total_num_rows; ++i) {
      MPConstraint* const ct = solver_->constraints_[i];
      DCHECK(!constraint_is_extracted(i));
      set_constraint_as_extracted(i, true);
      if (ct->coefficients_.size() > max_row_length) {
        max_row_length = ct->coefficients_.size();
      }
    }
    std::unique_ptr<SCIP_VAR*[]> vars(new SCIP_VAR*[max_row_length]);
    std::unique_ptr<double[]> coefs(new double[max_row_length]);
    // Add each new constraint.
    for (int i = last_constraint_index_; i < total_num_rows; ++i) {
      MPConstraint* const ct = solver_->constraints_[i];
      DCHECK(constraint_is_extracted(i));
      const int size = ct->coefficients_.size();
      int j = 0;
      for (const auto& entry : ct->coefficients_) {
        const int var_index = entry.first->index();
        DCHECK(variable_is_extracted(var_index));
        vars[j] = scip_variables_[var_index];
        coefs[j] = entry.second;
        j++;
      }
      SCIP_CONS* scip_constraint = nullptr;
      const bool is_lazy = ct->is_lazy();
      // See
      // http://scip.zib.de/doc/html/cons__linear_8h.php#aa7aed137a4130b35b168812414413481
      // for an explanation of the parameters.
      RETURN_IF_SCIP_ERROR(SCIPcreateConsLinear(
          scip_, &scip_constraint, ct->name().empty() ? "" : ct->name().c_str(),
          size, vars.get(), coefs.get(), ct->lb(), ct->ub(),
          !is_lazy,  // 'initial' parameter.
          true,      // 'separate' parameter.
          true,      // 'enforce' parameter.
          true,      // 'check' parameter.
          true,      // 'propagate' parameter.
          false,     // 'local' parameter.
          false,     // 'modifiable' parameter.
          false,     // 'dynamic' parameter.
          is_lazy,   // 'removable' parameter.
          false));   // 'stickingatnode' parameter.
      RETURN_IF_SCIP_ERROR(SCIPaddCons(scip_, scip_constraint));
      scip_constraints_.push_back(scip_constraint);
    }
  }
}

void SCIPInterface::ExtractObjective() {
  RETURN_IF_ALREADY_IN_ERROR_STATE;
  RETURN_IF_SCIP_ERROR(SCIPfreeTransform(scip_));
  // Linear objective: set objective coefficients for all variables (some might
  // have been modified).
  for (const auto& entry : solver_->objective_->coefficients_) {
    const int var_index = entry.first->index();
    const double obj_coef = entry.second;
    RETURN_IF_SCIP_ERROR(
        SCIPchgVarObj(scip_, scip_variables_[var_index], obj_coef));
  }

  // Constant term: change objective offset variable.
  RETURN_IF_SCIP_ERROR(SCIPchgVarObj(scip_, objective_offset_variable_,
                                     solver_->Objective().offset()));
}

#define RETURN_ABNORMAL_IF_BAD_STATUS             \
  do {                                            \
    if (!status_.ok()) {                          \
      LOG_IF(INFO, solver_->OutputIsEnabled())    \
          << "Invalid SCIP status: " << status_;  \
      return result_status_ = MPSolver::ABNORMAL; \
    }                                             \
  } while (false)

#define RETURN_ABNORMAL_IF_SCIP_ERROR(x) \
  do {                                   \
    RETURN_ABNORMAL_IF_BAD_STATUS;       \
    status_ = TO_STATUS(x);              \
    RETURN_ABNORMAL_IF_BAD_STATUS;       \
  } while (false);

MPSolver::ResultStatus SCIPInterface::Solve(const MPSolverParameters& param) {
  // "status_" may encode a variety of failure scenarios, many of which would
  // correspond to another MPResultStatus than ABNORMAL, but since SCIP is a
  // moving target, we use the most likely error code here (abnormalities,
  // often numeric), and rely on the user enabling output to see more details.
  RETURN_ABNORMAL_IF_BAD_STATUS;

  WallTimer timer;
  timer.Start();

  // Note that SCIP does not provide any incrementality.
  // TODO(user): Is that still true now (2018) ?
  if (param.GetIntegerParam(MPSolverParameters::INCREMENTALITY) ==
      MPSolverParameters::INCREMENTALITY_OFF) {
    Reset();
  }

  // Set log level.
  SCIPsetMessagehdlrQuiet(scip_, quiet_);

  // Special case if the model is empty since SCIP expects a non-empty model.
  if (solver_->variables_.empty() && solver_->constraints_.empty()) {
    sync_status_ = SOLUTION_SYNCHRONIZED;
    result_status_ = MPSolver::OPTIMAL;
    objective_value_ = solver_->Objective().offset();
    return result_status_;
  }

  ExtractModel();
  VLOG(1) << absl::StrFormat("Model built in %s.",
                             absl::FormatDuration(timer.GetDuration()));

  // Time limit.
  if (solver_->time_limit() != 0) {
    VLOG(1) << "Setting time limit = " << solver_->time_limit() << " ms.";
    RETURN_ABNORMAL_IF_SCIP_ERROR(
        SCIPsetRealParam(scip_, "limits/time", solver_->time_limit_in_secs()));
  } else {
    RETURN_ABNORMAL_IF_SCIP_ERROR(SCIPresetParam(scip_, "limits/time"));
  }

  // We first set our internal MPSolverParameters from param and then set any
  // user specified internal solver, ie. SCIP, parameters via
  // solver_specific_parameter_string_.
  // Default MPSolverParameters can override custom parameters (for example for
  // presolving) and therefore we apply MPSolverParameters first.
  SetParameters(param);
  solver_->SetSolverSpecificParametersAsString(
      solver_->solver_specific_parameter_string_);

  // Use the solution hint if any.
  if (!solver_->solution_hint_.empty()) {
    SCIP_SOL* solution;
    bool is_solution_partial = false;
    const int num_vars = solver_->variables_.size();
    if (solver_->solution_hint_.size() != num_vars) {
      // We start by creating an empty partial solution.
      RETURN_ABNORMAL_IF_SCIP_ERROR(
          SCIPcreatePartialSol(scip_, &solution, nullptr));
      is_solution_partial = true;
    } else {
      // We start by creating the all-zero solution.
      RETURN_ABNORMAL_IF_SCIP_ERROR(SCIPcreateSol(scip_, &solution, nullptr));
    }

    // The variable representing the objective offset should always be one!!
    // See CreateSCIP().
    RETURN_ABNORMAL_IF_SCIP_ERROR(
        SCIPsetSolVal(scip_, solution, objective_offset_variable_, 1.0));

    // Fill the other variables from the given solution hint.
    for (const std::pair<const MPVariable*, double>& p :
         solver_->solution_hint_) {
      RETURN_ABNORMAL_IF_SCIP_ERROR(SCIPsetSolVal(
          scip_, solution, scip_variables_[p.first->index()], p.second));
    }

    if (!is_solution_partial) {
      SCIP_Bool is_feasible;
      RETURN_ABNORMAL_IF_SCIP_ERROR(SCIPcheckSol(
          scip_, solution, /*printreason=*/false, /*completely=*/true,
          /*checkbounds=*/true, /*checkintegrality=*/true, /*checklprows=*/true,
          &is_feasible));
      VLOG(1) << "Solution hint is "
              << (is_feasible ? "FEASIBLE" : "INFEASIBLE");
    }

    // TODO(user): I more or less copied this from the SCIPreadSol() code that
    // reads a solution from a file. I am not sure what SCIPisTransformed() is
    // or what is the difference between the try and add version. In any case
    // this seems to always call SCIPaddSolFree() for now and it works.
    SCIP_Bool is_stored;
    if (!is_solution_partial && SCIPisTransformed(scip_)) {
      RETURN_ABNORMAL_IF_SCIP_ERROR(SCIPtrySolFree(
          scip_, &solution, /*printreason=*/false, /*completely=*/true,
          /*checkbounds=*/true, /*checkintegrality=*/true, /*checklprows=*/true,
          &is_stored));
    } else {
      RETURN_ABNORMAL_IF_SCIP_ERROR(
          SCIPaddSolFree(scip_, &solution, &is_stored));
    }
  }

  // Solve.
  timer.Restart();
  RETURN_ABNORMAL_IF_SCIP_ERROR(solver_->GetNumThreads() > 1
                                    ? SCIPsolveConcurrent(scip_)
                                    : SCIPsolve(scip_));
  VLOG(1) << absl::StrFormat("Solved in %s.",
                             absl::FormatDuration(timer.GetDuration()));

  // Get the results.
  SCIP_SOL* const solution = SCIPgetBestSol(scip_);
  if (solution != nullptr) {
    // If optimal or feasible solution is found.
    objective_value_ = SCIPgetSolOrigObj(scip_, solution);
    VLOG(1) << "objective=" << objective_value_;
    for (int i = 0; i < solver_->variables_.size(); ++i) {
      MPVariable* const var = solver_->variables_[i];
      const int var_index = var->index();
      const double val =
          SCIPgetSolVal(scip_, solution, scip_variables_[var_index]);
      var->set_solution_value(val);
      VLOG(3) << var->name() << "=" << val;
    }
  } else {
    VLOG(1) << "No feasible solution found.";
  }

  // Check the status: optimal, infeasible, etc.
  SCIP_STATUS scip_status = SCIPgetStatus(scip_);
  switch (scip_status) {
    case SCIP_STATUS_OPTIMAL:
      result_status_ = MPSolver::OPTIMAL;
      break;
    case SCIP_STATUS_GAPLIMIT:
      // To be consistent with the other solvers.
      result_status_ = MPSolver::OPTIMAL;
      break;
    case SCIP_STATUS_INFEASIBLE:
      result_status_ = MPSolver::INFEASIBLE;
      break;
    case SCIP_STATUS_UNBOUNDED:
      result_status_ = MPSolver::UNBOUNDED;
      break;
    case SCIP_STATUS_INFORUNBD:
      // TODO(user): We could introduce our own "infeasible or
      // unbounded" status.
      result_status_ = MPSolver::INFEASIBLE;
      break;
    default:
      if (solution != nullptr) {
        result_status_ = MPSolver::FEASIBLE;
      } else if (scip_status == SCIP_STATUS_TIMELIMIT) {
        result_status_ = MPSolver::NOT_SOLVED;
      } else {
        result_status_ = MPSolver::ABNORMAL;
      }
      break;
  }

  RETURN_ABNORMAL_IF_SCIP_ERROR(SCIPresetParams(scip_));

  sync_status_ = SOLUTION_SYNCHRONIZED;
  return result_status_;
}

int64 SCIPInterface::iterations() const {
  // NOTE(user): As of 2018-12 it doesn't run in the stubby server, and is
  // a specialized call, so it's ok to crash if the status is broken.
  if (!CheckSolutionIsSynchronized()) return kUnknownNumberOfIterations;
  return SCIPgetNLPIterations(scip_);
}

int64 SCIPInterface::nodes() const {
  // NOTE(user): Same story as iterations(): it's OK to crash here.
  if (!CheckSolutionIsSynchronized()) return kUnknownNumberOfNodes;
  // This is the total number of nodes used in the solve, potentially across
  // multiple branch-and-bound trees. Use limits/totalnodes (rather than
  // limits/nodes) to control this value.
  return SCIPgetNTotalNodes(scip_);
}

double SCIPInterface::best_objective_bound() const {
  // NOTE(user): Same story as iterations(): it's OK to crash here.
  if (!CheckSolutionIsSynchronized() || !CheckBestObjectiveBoundExists()) {
    return trivial_worst_objective_bound();
  }
  if (solver_->variables_.empty() && solver_->constraints_.empty()) {
    // Special case for empty model.
    return solver_->Objective().offset();
  } else {
    return SCIPgetDualbound(scip_);
  }
}

void SCIPInterface::SetParameters(const MPSolverParameters& param) {
  SetCommonParameters(param);
  SetMIPParameters(param);
}

void SCIPInterface::SetRelativeMipGap(double value) {
  // NOTE(user): We don't want to call RETURN_IF_ALREADY_IN_ERROR_STATE here,
  // because even if the solver is in an error state, the user might be setting
  // some parameters and then "restoring" the solver to a non-error state by
  // calling Reset(), which should *not* reset the parameters.
  // So we want the parameter-setting functions to be resistant to being in an
  // error state, essentially. What we do is:
  // - we call the parameter-setting function anyway (I'm assuming that SCIP
  //   won't crash even if we're in an error state. I did *not* verify this).
  // - if that call yielded an error *and* we weren't already in an error state,
  //   set the state to that error we just got.
  const auto status = TO_STATUS(SCIPsetRealParam(scip_, "limits/gap", value));
  if (status_.ok()) status_ = status;
}

void SCIPInterface::SetPrimalTolerance(double value) {
  // SCIP automatically updates numerics/lpfeastol if the primal tolerance is
  // tighter. Doing that it unconditionally logs this modification to stderr. By
  // setting numerics/lpfeastol first we avoid this unwanted log.
  double current_lpfeastol = 0.0;
  CHECK_EQ(SCIP_OKAY,
           SCIPgetRealParam(scip_, "numerics/lpfeastol", &current_lpfeastol));
  if (value < current_lpfeastol) {
    // See the NOTE on SetRelativeMipGap().
    const auto status =
        TO_STATUS(SCIPsetRealParam(scip_, "numerics/lpfeastol", value));
    if (status_.ok()) status_ = status;
  }
  // See the NOTE on SetRelativeMipGap().
  const auto status =
      TO_STATUS(SCIPsetRealParam(scip_, "numerics/feastol", value));
  if (status_.ok()) status_ = status;
}

void SCIPInterface::SetDualTolerance(double value) {
  const auto status =
      TO_STATUS(SCIPsetRealParam(scip_, "numerics/dualfeastol", value));
  if (status_.ok()) status_ = status;
}

void SCIPInterface::SetPresolveMode(int value) {
  // See the NOTE on SetRelativeMipGap().
  switch (value) {
    case MPSolverParameters::PRESOLVE_OFF: {
      const auto status =
          TO_STATUS(SCIPsetIntParam(scip_, "presolving/maxrounds", 0));
      if (status_.ok()) status_ = status;
      return;
    }
    case MPSolverParameters::PRESOLVE_ON: {
      const auto status =
          TO_STATUS(SCIPsetIntParam(scip_, "presolving/maxrounds", -1));
      if (status_.ok()) status_ = status;
      return;
    }
    default: {
      SetIntegerParamToUnsupportedValue(MPSolverParameters::PRESOLVE, value);
      return;
    }
  }
}

void SCIPInterface::SetScalingMode(int value) {
  SetUnsupportedIntegerParam(MPSolverParameters::SCALING);
}

// Only the root LP algorithm is set as setting the node LP to a
// non-default value rarely is beneficial. The node LP algorithm could
// be set as well with "lp/resolvealgorithm".
void SCIPInterface::SetLpAlgorithm(int value) {
  // See the NOTE on SetRelativeMipGap().
  switch (value) {
    case MPSolverParameters::DUAL: {
      const auto status =
          TO_STATUS(SCIPsetCharParam(scip_, "lp/initalgorithm", 'd'));
      if (status_.ok()) status_ = status;
      return;
    }
    case MPSolverParameters::PRIMAL: {
      const auto status =
          TO_STATUS(SCIPsetCharParam(scip_, "lp/initalgorithm", 'p'));
      if (status_.ok()) status_ = status;
      return;
    }
    case MPSolverParameters::BARRIER: {
      // Barrier with crossover.
      const auto status =
          TO_STATUS(SCIPsetCharParam(scip_, "lp/initalgorithm", 'p'));
      if (status_.ok()) status_ = status;
      return;
    }
    default: {
      SetIntegerParamToUnsupportedValue(MPSolverParameters::LP_ALGORITHM,
                                        value);
      return;
    }
  }
}

void SCIPInterface::SetUnsupportedIntegerParam(
    MPSolverParameters::IntegerParam param) {
  MPSolverInterface::SetUnsupportedIntegerParam(param);
  if (status_.ok()) {
    status_ = util::InvalidArgumentError(absl::StrFormat(
        "Tried to set unsupported integer parameter %d", param));
  }
}

void SCIPInterface::SetIntegerParamToUnsupportedValue(
    MPSolverParameters::IntegerParam param, int value) {
  MPSolverInterface::SetIntegerParamToUnsupportedValue(param, value);
  if (status_.ok()) {
    status_ = util::InvalidArgumentError(absl::StrFormat(
        "Tried to set integer parameter %d to unsupported value %d", param,
        value));
  }
}

util::Status SCIPInterface::SetNumThreads(int num_threads) {
  if (SetSolverSpecificParametersAsString(
          absl::StrFormat("parallel/maxnthreads = %d\n", num_threads))) {
    return util::OkStatus();
  }
  return util::InternalError(
      "Could not set parallel/maxnthreads, which may "
      "indicate that SCIP API has changed.");
}

bool SCIPInterface::ReadParameterFile(const std::string& filename) {
  return SCIPreadParams(scip_, filename.c_str()) == SCIP_OKAY;
}

std::string SCIPInterface::ValidFileExtensionForParameterFile() const {
  return ".set";
}

MPSolverInterface* BuildSCIPInterface(MPSolver* const solver) {
  return new SCIPInterface(solver);
}

}  // namespace operations_research
#endif  //  #if defined(USE_SCIP)

#undef TO_STATUS
#undef RETURN_IF_SCIP_ERROR
#undef RETURN_IF_ALREADY_IN_ERROR_STATE
#undef RETURN_ABNORMAL_IF_BAD_STATUS
#undef RETURN_ABNORMAL_IF_SCIP_ERROR
