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

#ifndef OR_TOOLS_SAT_CUTS_H_
#define OR_TOOLS_SAT_CUTS_H_

#include <utility>
#include <vector>

#include "ortools/base/int_type.h"
#include "ortools/sat/integer.h"
#include "ortools/sat/linear_constraint.h"
#include "ortools/sat/model.h"
#include "ortools/util/time_limit.h"

namespace operations_research {
namespace sat {

// A "cut" generator on a set of IntegerVariable.
//
// The generate_cuts() function will usually be called with the current LP
// optimal solution (but should work for any lp_values). Note that a
// CutGenerator should:
// - Only look at the lp_values positions that corresponds to its 'vars' or
//   their negation.
// - Only return cuts in term of the same variables or their negation.
struct CutGenerator {
  std::vector<IntegerVariable> vars;
  std::function<std::vector<LinearConstraint>(
      const gtl::ITIVector<IntegerVariable, double>& lp_values)>
      generate_cuts;
};

// If a variable is away from its upper bound by more than value 1.0, then it
// cannot be part of a cover that will violate the lp solution. This method
// returns a reduced constraint by removing such variables from the given
// constraint.
LinearConstraint GetPreprocessedLinearConstraint(
    const LinearConstraint& constraint,
    const gtl::ITIVector<IntegerVariable, double>& lp_values,
    const IntegerTrail& integer_trail);

// Returns true if sum of all the variables in the given constraint is less than
// or equal to constraint upper bound. This method assumes that all the
// coefficients are non negative.
bool ConstraintIsTriviallyTrue(const LinearConstraint& constraint,
                               const IntegerTrail& integer_trail);

// If the left variables in lp solution satisfies following inequality, we prove
// that there does not exist any knapsack cut which is violated by the solution.
// Let |Cmin| = smallest possible cover size.
// Let S = smallest (var_ub - lp_values[var]) first |Cmin| variables.
// Let cut lower bound = sum_(var in S)(var_ub - lp_values[var])
// For any cover,
// If cut lower bound >= 1
// ==> sum_(var in S)(var_ub - lp_values[var]) >= 1
// ==> sum_(var in cover)(var_ub - lp_values[var]) >= 1
// ==> The solution already satisfies cover. Since this is true for all covers,
// this method returns false in such cases.
// This method assumes that the constraint is preprocessed and has only non
// negative coefficients.
bool CanBeFilteredUsingCutLowerBound(
    const LinearConstraint& preprocessed_constraint,
    const gtl::ITIVector<IntegerVariable, double>& lp_values,
    const IntegerTrail& integer_trail);

// Struct to help compute upper bound for knapsack instance.
struct KnapsackItem {
  double profit;
  double weight;
  bool operator>(const KnapsackItem& other) const {
    return profit * other.weight > other.profit * weight;
  }
};

// Gets upper bound on profit for knapsack instance by solving the linear
// relaxation.
double GetKnapsackUpperBound(std::vector<KnapsackItem> items, double capacity);

// Returns true if the linear relaxation upper bound for the knapsack instance
// shows that this constraint cannot be used to form a cut. This method assumes
// that all the coefficients are non negative.
bool CanBeFilteredUsingKnapsackUpperBound(
    const LinearConstraint& constraint,
    const gtl::ITIVector<IntegerVariable, double>& lp_values,
    const IntegerTrail& integer_trail);

// Returns true if the given constraint passes all the filters described above.
// This method assumes that the constraint is preprocessed and has only non
// negative coefficients.
bool CanFormValidKnapsackCover(
    const LinearConstraint& preprocessed_constraint,
    const gtl::ITIVector<IntegerVariable, double>& lp_values,
    const IntegerTrail& integer_trail);

// Converts the given constraint into canonical knapsack form (described
// below) and adds it to 'knapsack_constraints'.
// Canonical knapsack form:
//  - Constraint has finite upper bound.
//  - All coefficients are positive.
// For constraint with finite lower bound, this method also adds the negation of
// the given constraint after converting it to canonical knapsack form.
void ConvertToKnapsackForm(const LinearConstraint& constraint,
                           std::vector<LinearConstraint>* knapsack_constraints);

// Returns true if the cut is lifted. Lifting procedure is described below.
//
// First we decide a lifting sequence for the binary variables which are not
// already in cut. We lift the cut for each lifting candidate one by one.
//
// Given the original constraint where the lifting candidate is fixed to one, we
// compute the maximum value the cut can take and still be feasible using a
// knapsack problem. We can then lift the variable in the cut using the
// difference between the cut upper bound and this maximum value.
bool LiftKnapsackCut(
    const LinearConstraint& constraint,
    const gtl::ITIVector<IntegerVariable, double>& lp_values,
    const std::vector<IntegerValue>& cut_vars_original_coefficients,
    const IntegerTrail& integer_trail, TimeLimit* time_limit,
    LinearConstraint* cut);

// A cut generator that creates knpasack cover cuts.
//
// For a constraint of type
// \sum_{i=1..n}(a_i * x_i) <= b
// where x_i are integer variables with upper bound u_i, a cover of size k is a
// subset C of {1 , .. , n} such that \sum_{c \in C}(a_c * u_c) > b.
//
// A knapsack cover cut is a constraint of the form
// \sum_{c \in C}(u_c - x_c) >= 1
// which is equivalent to \sum_{c \in C}(x_c) <= \sum_{c \in C}(u_c) - 1.
// In other words, in a feasible solution, at least some of the variables do
// not take their maximum value.
//
// If all x_i are binary variables then the cover cut becomes
// \sum_{c \in C}(x_c) <= |C| - 1.
//
// The major difficulty for generating Knapsack cover cuts is finding a minimal
// cover set C that cut a given floating point solution. There are many ways to
// heuristically generate the cover but the following method that uses a
// solution of the LP relaxation of the constraint works the best.
//
// Look at a given linear relaxation solution for the integer problem x'
// and try to solve the following knapsack problem:
//   Minimize \sum_{i=1..n}(z_i * (u_i - x_i')),
//   such that \sum_{i=1..n}(a_i * u_i * z_i) > b,
// where z_i is a binary decision variable and x_i' are values of the variables
// in the given relaxation solution x'. If the objective of the optimal solution
// of this problem is less than 1, this algorithm does not generate any cuts.
// Otherwise, it adds a knapsack cover cut in the form
//   \sum_{i=1..n}(z_i' * x_i) <= cb,
// where z_i' is the value of z_i in the optimal solution of the above
// problem and cb is the upper bound for the cut constraint. Note that the above
// problem can be converted into a standard kanpsack form by replacing z_i by 1
// - y_i. In that case the problem becomes
//   Maximize \sum_{i=1..n}((u_i - x_i') * (y_i - 1)),
//   such that
//     \sum_{i=1..n}(a_i * u_i * y_i) <= \sum_{i=1..n}(a_i * u_i) - b - 1.
//
// Solving this knapsack instance would help us find the smallest cover with
// maximum LP violation.
//
// Cut strengthning:
// Let lambda = \sum_{c \in C}(a_c * u_c) - b and max_coeff = \max_{c
// \in C}(a_c), then cut can be strengthened as
//   \sum_{c \in C}(u_c - x_c) >= ceil(lambda / max_coeff)
//
// For further information about knapsack cover cuts see
// A. Atamtürk, Cover and Pack Inequalities for (Mixed) Integer Programming
// Annals of Operations Research Volume 139, Issue 1 , pp 21-38, 2005.
// TODO(user): Implement cut lifting.
CutGenerator CreateKnapsackCoverCutGenerator(
    const std::vector<LinearConstraint>& base_constraints,
    const std::vector<IntegerVariable>& vars, Model* model);

}  // namespace sat
}  // namespace operations_research

#endif  // OR_TOOLS_SAT_CUTS_H_
