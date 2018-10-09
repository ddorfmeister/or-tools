BASE_DEPS = \
 $(SRC_DIR)/ortools/base/accurate_sum.h \
 $(SRC_DIR)/ortools/base/adjustable_priority_queue.h \
 $(SRC_DIR)/ortools/base/adjustable_priority_queue-inl.h \
 $(SRC_DIR)/ortools/base/base_export.h \
 $(SRC_DIR)/ortools/base/basictypes.h \
 $(SRC_DIR)/ortools/base/bitmap.h \
 $(SRC_DIR)/ortools/base/callback.h \
 $(SRC_DIR)/ortools/base/canonical_errors.h \
 $(SRC_DIR)/ortools/base/casts.h \
 $(SRC_DIR)/ortools/base/cleanup.h \
 $(SRC_DIR)/ortools/base/commandlineflags.h \
 $(SRC_DIR)/ortools/base/dynamic_library.h \
 $(SRC_DIR)/ortools/base/encodingutils.h \
 $(SRC_DIR)/ortools/base/file.h \
 $(SRC_DIR)/ortools/base/filelineiter.h \
 $(SRC_DIR)/ortools/base/hash.h \
 $(SRC_DIR)/ortools/base/inlined_vector.h \
 $(SRC_DIR)/ortools/base/integral_types.h \
 $(SRC_DIR)/ortools/base/int_type.h \
 $(SRC_DIR)/ortools/base/int_type_indexed_vector.h \
 $(SRC_DIR)/ortools/base/iterator_adaptors.h \
 $(SRC_DIR)/ortools/base/jniutil.h \
 $(SRC_DIR)/ortools/base/join.h \
 $(SRC_DIR)/ortools/base/logging.h \
 $(SRC_DIR)/ortools/base/macros.h \
 $(SRC_DIR)/ortools/base/map_util.h \
 $(SRC_DIR)/ortools/base/match.h \
 $(SRC_DIR)/ortools/base/mathutil.h \
 $(SRC_DIR)/ortools/base/memory.h \
 $(SRC_DIR)/ortools/base/murmur.h \
 $(SRC_DIR)/ortools/base/mutex.h \
 $(SRC_DIR)/ortools/base/notification.h \
 $(SRC_DIR)/ortools/base/numbers.h \
 $(SRC_DIR)/ortools/base/optional.h \
 $(SRC_DIR)/ortools/base/port.h \
 $(SRC_DIR)/ortools/base/protoutil.h \
 $(SRC_DIR)/ortools/base/ptr_util.h \
 $(SRC_DIR)/ortools/base/python-swig.h \
 $(SRC_DIR)/ortools/base/random.h \
 $(SRC_DIR)/ortools/base/recordio.h \
 $(SRC_DIR)/ortools/base/small_map.h \
 $(SRC_DIR)/ortools/base/small_ordered_set.h \
 $(SRC_DIR)/ortools/base/span.h \
 $(SRC_DIR)/ortools/base/sparsetable.h \
 $(SRC_DIR)/ortools/base/split.h \
 $(SRC_DIR)/ortools/base/status.h \
 $(SRC_DIR)/ortools/base/statusor.h \
 $(SRC_DIR)/ortools/base/stl_util.h \
 $(SRC_DIR)/ortools/base/stringpiece_utils.h \
 $(SRC_DIR)/ortools/base/stringprintf.h \
 $(SRC_DIR)/ortools/base/string_view.h \
 $(SRC_DIR)/ortools/base/strtoint.h \
 $(SRC_DIR)/ortools/base/strutil.h \
 $(SRC_DIR)/ortools/base/synchronization.h \
 $(SRC_DIR)/ortools/base/sysinfo.h \
 $(SRC_DIR)/ortools/base/thorough_hash.h \
 $(SRC_DIR)/ortools/base/threadpool.h \
 $(SRC_DIR)/ortools/base/timer.h \
 $(SRC_DIR)/ortools/base/time_support.h \
 $(SRC_DIR)/ortools/base/typeid.h

BASE_LIB_OBJS = \
 $(OBJ_DIR)/base/bitmap.$O \
 $(OBJ_DIR)/base/callback.$O \
 $(OBJ_DIR)/base/dynamic_library.$O \
 $(OBJ_DIR)/base/file.$O \
 $(OBJ_DIR)/base/join.$O \
 $(OBJ_DIR)/base/mutex.$O \
 $(OBJ_DIR)/base/notification.$O \
 $(OBJ_DIR)/base/numbers.$O \
 $(OBJ_DIR)/base/random.$O \
 $(OBJ_DIR)/base/recordio.$O \
 $(OBJ_DIR)/base/split.$O \
 $(OBJ_DIR)/base/string_view.$O \
 $(OBJ_DIR)/base/stringprintf.$O \
 $(OBJ_DIR)/base/sysinfo.$O \
 $(OBJ_DIR)/base/threadpool.$O \
 $(OBJ_DIR)/base/time_support.$O \
 $(OBJ_DIR)/base/timer.$O

objs/base/bitmap.$O: ortools/base/bitmap.cc ortools/base/bitmap.h \
 ortools/base/basictypes.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/macros.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Sbitmap.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Sbitmap.$O

objs/base/callback.$O: ortools/base/callback.cc ortools/base/callback.h \
 ortools/base/logging.h ortools/base/integral_types.h \
 ortools/base/macros.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Scallback.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Scallback.$O

objs/base/dynamic_library.$O: ortools/base/dynamic_library.cc \
 ortools/base/dynamic_library.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Sdynamic_library.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Sdynamic_library.$O

 | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Sfile.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Sfile.$O

objs/base/join.$O: ortools/base/join.cc ortools/base/join.h \
 ortools/base/basictypes.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/macros.h ortools/base/string_view.h \
 ortools/base/stringprintf.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Sjoin.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Sjoin.$O

 | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Smutex.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Smutex.$O

 | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Snotification.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Snotification.$O

objs/base/numbers.$O: ortools/base/numbers.cc ortools/base/numbers.h \
 ortools/base/integral_types.h ortools/base/join.h \
 ortools/base/basictypes.h ortools/base/logging.h ortools/base/macros.h \
 ortools/base/string_view.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Snumbers.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Snumbers.$O

 | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Srandom.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Srandom.$O

 | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Srecordio.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Srecordio.$O

objs/base/split.$O: ortools/base/split.cc ortools/base/split.h \
 ortools/base/integral_types.h ortools/base/logging.h \
 ortools/base/macros.h ortools/base/string_view.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Ssplit.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Ssplit.$O

objs/base/string_view.$O: ortools/base/string_view.cc \
 ortools/base/string_view.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Sstring_view.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Sstring_view.$O

objs/base/stringprintf.$O: ortools/base/stringprintf.cc \
 ortools/base/stringprintf.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Sstringprintf.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Sstringprintf.$O

objs/base/sysinfo.$O: ortools/base/sysinfo.cc ortools/base/sysinfo.h \
 ortools/base/basictypes.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/macros.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Ssysinfo.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Ssysinfo.$O

 | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Sthreadpool.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Sthreadpool.$O

objs/base/time_support.$O: ortools/base/time_support.cc \
 ortools/base/time_support.h ortools/base/integral_types.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Stime_support.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Stime_support.$O

objs/base/timer.$O: ortools/base/timer.cc ortools/base/timer.h \
 ortools/base/basictypes.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/macros.h ortools/base/time_support.h | $(OBJ_DIR)/base
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbase$Stimer.cc $(OBJ_OUT)$(OBJ_DIR)$Sbase$Stimer.$O

PORT_DEPS = \
 $(SRC_DIR)/ortools/port/file.h \
 $(SRC_DIR)/ortools/port/proto_utils.h \
 $(SRC_DIR)/ortools/port/sysinfo.h \
 $(SRC_DIR)/ortools/port/utf8.h

PORT_LIB_OBJS = \
 $(OBJ_DIR)/port/file_nonport.$O \
 $(OBJ_DIR)/port/sysinfo_nonport.$O

 | $(OBJ_DIR)/port
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sport$Sfile_nonport.cc $(OBJ_OUT)$(OBJ_DIR)$Sport$Sfile_nonport.$O

objs/port/sysinfo_nonport.$O: ortools/port/sysinfo_nonport.cc \
 ortools/port/sysinfo.h ortools/base/integral_types.h \
 ortools/base/sysinfo.h ortools/base/basictypes.h ortools/base/logging.h \
 ortools/base/macros.h | $(OBJ_DIR)/port
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sport$Ssysinfo_nonport.cc $(OBJ_OUT)$(OBJ_DIR)$Sport$Ssysinfo_nonport.$O

UTIL_DEPS = \
 $(SRC_DIR)/ortools/util/affine_relation.h \
 $(SRC_DIR)/ortools/util/bitset.h \
 $(SRC_DIR)/ortools/util/cached_log.h \
 $(SRC_DIR)/ortools/util/file_util.h \
 $(SRC_DIR)/ortools/util/fp_utils.h \
 $(SRC_DIR)/ortools/util/functions_swig_helpers.h \
 $(SRC_DIR)/ortools/util/functions_swig_test_helpers.h \
 $(SRC_DIR)/ortools/util/graph_export.h \
 $(SRC_DIR)/ortools/util/integer_pq.h \
 $(SRC_DIR)/ortools/util/monoid_operation_tree.h \
 $(SRC_DIR)/ortools/util/permutation.h \
 $(SRC_DIR)/ortools/util/piecewise_linear_function.h \
 $(SRC_DIR)/ortools/util/proto_tools.h \
 $(SRC_DIR)/ortools/util/random_engine.h \
 $(SRC_DIR)/ortools/util/range_minimum_query.h \
 $(SRC_DIR)/ortools/util/range_query_function.h \
 $(SRC_DIR)/ortools/util/rational_approximation.h \
 $(SRC_DIR)/ortools/util/return_macros.h \
 $(SRC_DIR)/ortools/util/rev.h \
 $(SRC_DIR)/ortools/util/running_stat.h \
 $(SRC_DIR)/ortools/util/saturated_arithmetic.h \
 $(SRC_DIR)/ortools/util/sigint.h \
 $(SRC_DIR)/ortools/util/sorted_interval_list.h \
 $(SRC_DIR)/ortools/util/sort.h \
 $(SRC_DIR)/ortools/util/stats.h \
 $(SRC_DIR)/ortools/util/string_array.h \
 $(SRC_DIR)/ortools/util/time_limit.h \
 $(SRC_DIR)/ortools/util/tuple_set.h \
 $(SRC_DIR)/ortools/util/vector_map.h \
 $(SRC_DIR)/ortools/util/vector_or_function.h \
 $(SRC_DIR)/ortools/util/xml_helper.h \
 $(SRC_DIR)/ortools/util/zvector.h \
 $(GEN_DIR)/ortools/util/optional_boolean.pb.h

UTIL_LIB_OBJS = \
 $(OBJ_DIR)/util/bitset.$O \
 $(OBJ_DIR)/util/cached_log.$O \
 $(OBJ_DIR)/util/file_util.$O \
 $(OBJ_DIR)/util/fp_utils.$O \
 $(OBJ_DIR)/util/graph_export.$O \
 $(OBJ_DIR)/util/piecewise_linear_function.$O \
 $(OBJ_DIR)/util/proto_tools.$O \
 $(OBJ_DIR)/util/range_query_function.$O \
 $(OBJ_DIR)/util/rational_approximation.$O \
 $(OBJ_DIR)/util/sigint.$O \
 $(OBJ_DIR)/util/sorted_interval_list.$O \
 $(OBJ_DIR)/util/stats.$O \
 $(OBJ_DIR)/util/time_limit.$O \
 $(OBJ_DIR)/util/xml_helper.$O \
 $(OBJ_DIR)/util/optional_boolean.pb.$O

objs/util/bitset.$O: ortools/util/bitset.cc ortools/util/bitset.h \
 ortools/base/basictypes.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/macros.h \
 ortools/base/commandlineflags.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Sbitset.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Sbitset.$O

objs/util/cached_log.$O: ortools/util/cached_log.cc \
 ortools/util/cached_log.h ortools/base/basictypes.h \
 ortools/base/integral_types.h ortools/base/logging.h \
 ortools/base/macros.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Scached_log.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Scached_log.$O

 | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Sfile_util.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Sfile_util.$O

objs/util/fp_utils.$O: ortools/util/fp_utils.cc ortools/util/fp_utils.h \
 ortools/base/logging.h ortools/base/integral_types.h \
 ortools/base/macros.h ortools/util/bitset.h ortools/base/basictypes.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Sfp_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Sfp_utils.$O

 | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Sgraph_export.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Sgraph_export.$O

objs/util/piecewise_linear_function.$O: \
 ortools/util/piecewise_linear_function.cc \
 ortools/util/piecewise_linear_function.h ortools/base/basictypes.h \
 ortools/base/integral_types.h ortools/base/logging.h \
 ortools/base/macros.h ortools/util/saturated_arithmetic.h \
 ortools/base/casts.h ortools/util/bitset.h ortools/base/stringprintf.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Spiecewise_linear_function.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Spiecewise_linear_function.$O

 | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Sproto_tools.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Sproto_tools.$O

objs/util/range_query_function.$O: ortools/util/range_query_function.cc \
 ortools/util/range_query_function.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/macros.h \
 ortools/util/range_minimum_query.h ortools/util/bitset.h \
 ortools/base/basictypes.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Srange_query_function.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Srange_query_function.$O

objs/util/rational_approximation.$O: \
 ortools/util/rational_approximation.cc \
 ortools/util/rational_approximation.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/macros.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Srational_approximation.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Srational_approximation.$O

objs/util/sigint.$O: ortools/util/sigint.cc ortools/util/sigint.h \
 ortools/base/logging.h ortools/base/integral_types.h \
 ortools/base/macros.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Ssigint.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Ssigint.$O

 | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Ssorted_interval_list.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Ssorted_interval_list.$O

objs/util/stats.$O: ortools/util/stats.cc ortools/util/stats.h \
 ortools/base/timer.h ortools/base/basictypes.h \
 ortools/base/integral_types.h ortools/base/logging.h \
 ortools/base/macros.h ortools/base/time_support.h \
 ortools/base/stringprintf.h ortools/base/stl_util.h \
 ortools/port/sysinfo.h ortools/port/utf8.h ortools/base/encodingutils.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Sstats.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Sstats.$O

 | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Stime_limit.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Stime_limit.$O

objs/util/xml_helper.$O: ortools/util/xml_helper.cc \
 ortools/util/xml_helper.h ortools/base/macros.h ortools/base/join.h \
 ortools/base/basictypes.h ortools/base/integral_types.h \
 ortools/base/logging.h ortools/base/string_view.h \
 ortools/base/stringprintf.h ortools/base/strutil.h | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sutil$Sxml_helper.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Sxml_helper.$O

ortools/util/optional_boolean.proto: ;

$(GEN_DIR)/ortools/util/optional_boolean.pb.cc: \
 $(SRC_DIR)/ortools/util/optional_boolean.proto | $(GEN_DIR)/ortools/util
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/util/optional_boolean.proto

$(GEN_DIR)/ortools/util/optional_boolean.pb.h: \
 $(GEN_DIR)/ortools/util/optional_boolean.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sutil$Soptional_boolean.pb.h

$(OBJ_DIR)/util/optional_boolean.pb.$O: \
 $(GEN_DIR)/ortools/util/optional_boolean.pb.cc | $(OBJ_DIR)/util
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sutil$Soptional_boolean.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sutil$Soptional_boolean.pb.$O

DATA_DEPS = \
 $(SRC_DIR)/ortools/data/jobshop_scheduling_parser.h \
 $(SRC_DIR)/ortools/data/rcpsp_parser.h \
 $(SRC_DIR)/ortools/data/set_covering_data.h \
 $(SRC_DIR)/ortools/data/set_covering_parser.h \
 $(GEN_DIR)/ortools/data/jobshop_scheduling.pb.h \
 $(GEN_DIR)/ortools/data/rcpsp.pb.h

DATA_LIB_OBJS = \
 $(OBJ_DIR)/data/jobshop_scheduling_parser.$O \
 $(OBJ_DIR)/data/rcpsp_parser.$O \
 $(OBJ_DIR)/data/set_covering_data.$O \
 $(OBJ_DIR)/data/set_covering_parser.$O \
 $(OBJ_DIR)/data/jobshop_scheduling.pb.$O \
 $(OBJ_DIR)/data/rcpsp.pb.$O

 | $(OBJ_DIR)/data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sdata$Sjobshop_scheduling_parser.cc $(OBJ_OUT)$(OBJ_DIR)$Sdata$Sjobshop_scheduling_parser.$O

 | $(OBJ_DIR)/data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sdata$Srcpsp_parser.cc $(OBJ_OUT)$(OBJ_DIR)$Sdata$Srcpsp_parser.$O

objs/data/set_covering_data.$O: ortools/data/set_covering_data.cc \
 ortools/data/set_covering_data.h ortools/base/integral_types.h | $(OBJ_DIR)/data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sdata$Sset_covering_data.cc $(OBJ_OUT)$(OBJ_DIR)$Sdata$Sset_covering_data.$O

 | $(OBJ_DIR)/data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sdata$Sset_covering_parser.cc $(OBJ_OUT)$(OBJ_DIR)$Sdata$Sset_covering_parser.$O

ortools/data/jobshop_scheduling.proto: ;

$(GEN_DIR)/ortools/data/jobshop_scheduling.pb.cc: \
 $(SRC_DIR)/ortools/data/jobshop_scheduling.proto | $(GEN_DIR)/ortools/data
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/data/jobshop_scheduling.proto

$(GEN_DIR)/ortools/data/jobshop_scheduling.pb.h: \
 $(GEN_DIR)/ortools/data/jobshop_scheduling.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sdata$Sjobshop_scheduling.pb.h

$(OBJ_DIR)/data/jobshop_scheduling.pb.$O: \
 $(GEN_DIR)/ortools/data/jobshop_scheduling.pb.cc | $(OBJ_DIR)/data
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sdata$Sjobshop_scheduling.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sdata$Sjobshop_scheduling.pb.$O

ortools/data/rcpsp.proto: ;

$(GEN_DIR)/ortools/data/rcpsp.pb.cc: \
 $(SRC_DIR)/ortools/data/rcpsp.proto | $(GEN_DIR)/ortools/data
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/data/rcpsp.proto

$(GEN_DIR)/ortools/data/rcpsp.pb.h: \
 $(GEN_DIR)/ortools/data/rcpsp.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sdata$Srcpsp.pb.h

$(OBJ_DIR)/data/rcpsp.pb.$O: \
 $(GEN_DIR)/ortools/data/rcpsp.pb.cc | $(OBJ_DIR)/data
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sdata$Srcpsp.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sdata$Srcpsp.pb.$O

LP_DATA_DEPS = \
 $(SRC_DIR)/ortools/lp_data/lp_data.h \
 $(SRC_DIR)/ortools/lp_data/lp_data_utils.h \
 $(SRC_DIR)/ortools/lp_data/lp_decomposer.h \
 $(SRC_DIR)/ortools/lp_data/lp_print_utils.h \
 $(SRC_DIR)/ortools/lp_data/lp_types.h \
 $(SRC_DIR)/ortools/lp_data/lp_utils.h \
 $(SRC_DIR)/ortools/lp_data/matrix_scaler.h \
 $(SRC_DIR)/ortools/lp_data/matrix_utils.h \
 $(SRC_DIR)/ortools/lp_data/model_reader.h \
 $(SRC_DIR)/ortools/lp_data/mps_reader.h \
 $(SRC_DIR)/ortools/lp_data/permutation.h \
 $(SRC_DIR)/ortools/lp_data/proto_utils.h \
 $(SRC_DIR)/ortools/lp_data/sparse_column.h \
 $(SRC_DIR)/ortools/lp_data/sparse.h \
 $(SRC_DIR)/ortools/lp_data/sparse_row.h \
 $(SRC_DIR)/ortools/lp_data/sparse_vector.h

LP_DATA_LIB_OBJS = \
 $(OBJ_DIR)/lp_data/lp_data.$O \
 $(OBJ_DIR)/lp_data/lp_data_utils.$O \
 $(OBJ_DIR)/lp_data/lp_decomposer.$O \
 $(OBJ_DIR)/lp_data/lp_print_utils.$O \
 $(OBJ_DIR)/lp_data/lp_types.$O \
 $(OBJ_DIR)/lp_data/lp_utils.$O \
 $(OBJ_DIR)/lp_data/matrix_scaler.$O \
 $(OBJ_DIR)/lp_data/matrix_utils.$O \
 $(OBJ_DIR)/lp_data/model_reader.$O \
 $(OBJ_DIR)/lp_data/mps_reader.$O \
 $(OBJ_DIR)/lp_data/proto_utils.$O \
 $(OBJ_DIR)/lp_data/sparse.$O \
 $(OBJ_DIR)/lp_data/sparse_column.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Slp_data.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Slp_data.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Slp_data_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Slp_data_utils.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Slp_decomposer.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Slp_decomposer.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Slp_print_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Slp_print_utils.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Slp_types.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Slp_types.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Slp_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Slp_utils.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Smatrix_scaler.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Smatrix_scaler.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Smatrix_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Smatrix_utils.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Smodel_reader.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Smodel_reader.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Smps_reader.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Smps_reader.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Sproto_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Sproto_utils.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Ssparse.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Ssparse.$O

 | $(OBJ_DIR)/lp_data
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slp_data$Ssparse_column.cc $(OBJ_OUT)$(OBJ_DIR)$Slp_data$Ssparse_column.$O

GLOP_DEPS = \
 $(SRC_DIR)/ortools/glop/basis_representation.h \
 $(SRC_DIR)/ortools/glop/dual_edge_norms.h \
 $(SRC_DIR)/ortools/glop/entering_variable.h \
 $(SRC_DIR)/ortools/glop/initial_basis.h \
 $(SRC_DIR)/ortools/glop/lp_solver.h \
 $(SRC_DIR)/ortools/glop/lu_factorization.h \
 $(SRC_DIR)/ortools/glop/markowitz.h \
 $(SRC_DIR)/ortools/glop/preprocessor.h \
 $(SRC_DIR)/ortools/glop/primal_edge_norms.h \
 $(SRC_DIR)/ortools/glop/rank_one_update.h \
 $(SRC_DIR)/ortools/glop/reduced_costs.h \
 $(SRC_DIR)/ortools/glop/revised_simplex.h \
 $(SRC_DIR)/ortools/glop/status.h \
 $(SRC_DIR)/ortools/glop/update_row.h \
 $(SRC_DIR)/ortools/glop/variables_info.h \
 $(SRC_DIR)/ortools/glop/variable_values.h \
 $(GEN_DIR)/ortools/glop/parameters.pb.h

GLOP_LIB_OBJS = \
 $(OBJ_DIR)/glop/basis_representation.$O \
 $(OBJ_DIR)/glop/dual_edge_norms.$O \
 $(OBJ_DIR)/glop/entering_variable.$O \
 $(OBJ_DIR)/glop/initial_basis.$O \
 $(OBJ_DIR)/glop/lp_solver.$O \
 $(OBJ_DIR)/glop/lu_factorization.$O \
 $(OBJ_DIR)/glop/markowitz.$O \
 $(OBJ_DIR)/glop/preprocessor.$O \
 $(OBJ_DIR)/glop/primal_edge_norms.$O \
 $(OBJ_DIR)/glop/reduced_costs.$O \
 $(OBJ_DIR)/glop/revised_simplex.$O \
 $(OBJ_DIR)/glop/status.$O \
 $(OBJ_DIR)/glop/update_row.$O \
 $(OBJ_DIR)/glop/variable_values.$O \
 $(OBJ_DIR)/glop/variables_info.$O \
 $(OBJ_DIR)/glop/parameters.pb.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Sbasis_representation.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sbasis_representation.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Sdual_edge_norms.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sdual_edge_norms.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Sentering_variable.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sentering_variable.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Sinitial_basis.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sinitial_basis.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Slp_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Slp_solver.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Slu_factorization.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Slu_factorization.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Smarkowitz.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Smarkowitz.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Spreprocessor.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Spreprocessor.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Sprimal_edge_norms.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sprimal_edge_norms.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Sreduced_costs.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sreduced_costs.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Srevised_simplex.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Srevised_simplex.$O

objs/glop/status.$O: ortools/glop/status.cc ortools/glop/status.h \
 ortools/base/port.h ortools/base/logging.h ortools/base/integral_types.h \
 ortools/base/macros.h | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Sstatus.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sstatus.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Supdate_row.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Supdate_row.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Svariable_values.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Svariable_values.$O

 | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sglop$Svariables_info.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Svariables_info.$O

ortools/glop/parameters.proto: ;

$(GEN_DIR)/ortools/glop/parameters.pb.cc: \
 $(SRC_DIR)/ortools/glop/parameters.proto | $(GEN_DIR)/ortools/glop
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/glop/parameters.proto

$(GEN_DIR)/ortools/glop/parameters.pb.h: \
 $(GEN_DIR)/ortools/glop/parameters.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sglop$Sparameters.pb.h

$(OBJ_DIR)/glop/parameters.pb.$O: \
 $(GEN_DIR)/ortools/glop/parameters.pb.cc | $(OBJ_DIR)/glop
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sglop$Sparameters.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sglop$Sparameters.pb.$O

GRAPH_DEPS = \
 $(SRC_DIR)/ortools/graph/assignment.h \
 $(SRC_DIR)/ortools/graph/christofides.h \
 $(SRC_DIR)/ortools/graph/cliques.h \
 $(SRC_DIR)/ortools/graph/connected_components.h \
 $(SRC_DIR)/ortools/graph/connectivity.h \
 $(SRC_DIR)/ortools/graph/ebert_graph.h \
 $(SRC_DIR)/ortools/graph/eulerian_path.h \
 $(SRC_DIR)/ortools/graph/graph.h \
 $(SRC_DIR)/ortools/graph/graphs.h \
 $(SRC_DIR)/ortools/graph/hamiltonian_path.h \
 $(SRC_DIR)/ortools/graph/io.h \
 $(SRC_DIR)/ortools/graph/iterators.h \
 $(SRC_DIR)/ortools/graph/linear_assignment.h \
 $(SRC_DIR)/ortools/graph/max_flow.h \
 $(SRC_DIR)/ortools/graph/min_cost_flow.h \
 $(SRC_DIR)/ortools/graph/minimum_spanning_tree.h \
 $(SRC_DIR)/ortools/graph/one_tree_lower_bound.h \
 $(SRC_DIR)/ortools/graph/shortestpaths.h \
 $(SRC_DIR)/ortools/graph/strongly_connected_components.h \
 $(SRC_DIR)/ortools/graph/util.h \
 $(GEN_DIR)/ortools/graph/flow_problem.pb.h

GRAPH_LIB_OBJS = \
 $(OBJ_DIR)/graph/assignment.$O \
 $(OBJ_DIR)/graph/astar.$O \
 $(OBJ_DIR)/graph/bellman_ford.$O \
 $(OBJ_DIR)/graph/cliques.$O \
 $(OBJ_DIR)/graph/connected_components.$O \
 $(OBJ_DIR)/graph/dijkstra.$O \
 $(OBJ_DIR)/graph/linear_assignment.$O \
 $(OBJ_DIR)/graph/max_flow.$O \
 $(OBJ_DIR)/graph/min_cost_flow.$O \
 $(OBJ_DIR)/graph/shortestpaths.$O \
 $(OBJ_DIR)/graph/util.$O \
 $(OBJ_DIR)/graph/flow_problem.pb.$O

objs/graph/assignment.$O: ortools/graph/assignment.cc \
 ortools/graph/assignment.h ortools/graph/ebert_graph.h \
 ortools/base/integral_types.h ortools/base/join.h \
 ortools/base/basictypes.h ortools/base/logging.h ortools/base/macros.h \
 ortools/base/string_view.h ortools/base/stringprintf.h \
 ortools/util/permutation.h ortools/util/zvector.h \
 ortools/base/commandlineflags.h ortools/graph/linear_assignment.h | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Sassignment.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sassignment.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Sastar.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sastar.$O

objs/graph/bellman_ford.$O: ortools/graph/bellman_ford.cc \
 ortools/base/integral_types.h | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Sbellman_ford.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sbellman_ford.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Scliques.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Scliques.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Sconnected_components.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sconnected_components.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Sdijkstra.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sdijkstra.$O

objs/graph/linear_assignment.$O: ortools/graph/linear_assignment.cc \
 ortools/graph/linear_assignment.h ortools/base/commandlineflags.h \
 ortools/base/integral_types.h ortools/base/logging.h \
 ortools/base/macros.h ortools/base/stringprintf.h \
 ortools/graph/ebert_graph.h ortools/base/join.h \
 ortools/base/basictypes.h ortools/base/string_view.h \
 ortools/util/permutation.h ortools/util/zvector.h | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Slinear_assignment.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Slinear_assignment.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Smax_flow.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Smax_flow.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Smin_cost_flow.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Smin_cost_flow.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Sshortestpaths.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sshortestpaths.$O

 | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sgraph$Sutil.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sutil.$O

ortools/graph/flow_problem.proto: ;

$(GEN_DIR)/ortools/graph/flow_problem.pb.cc: \
 $(SRC_DIR)/ortools/graph/flow_problem.proto | $(GEN_DIR)/ortools/graph
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/graph/flow_problem.proto

$(GEN_DIR)/ortools/graph/flow_problem.pb.h: \
 $(GEN_DIR)/ortools/graph/flow_problem.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sgraph$Sflow_problem.pb.h

$(OBJ_DIR)/graph/flow_problem.pb.$O: \
 $(GEN_DIR)/ortools/graph/flow_problem.pb.cc | $(OBJ_DIR)/graph
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sgraph$Sflow_problem.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sgraph$Sflow_problem.pb.$O

ALGORITHMS_DEPS = \
 $(SRC_DIR)/ortools/algorithms/dense_doubly_linked_list.h \
 $(SRC_DIR)/ortools/algorithms/dynamic_partition.h \
 $(SRC_DIR)/ortools/algorithms/dynamic_permutation.h \
 $(SRC_DIR)/ortools/algorithms/find_graph_symmetries.h \
 $(SRC_DIR)/ortools/algorithms/hungarian.h \
 $(SRC_DIR)/ortools/algorithms/knapsack_solver.h \
 $(SRC_DIR)/ortools/algorithms/sparse_permutation.h

ALGORITHMS_LIB_OBJS = \
 $(OBJ_DIR)/algorithms/dynamic_partition.$O \
 $(OBJ_DIR)/algorithms/dynamic_permutation.$O \
 $(OBJ_DIR)/algorithms/find_graph_symmetries.$O \
 $(OBJ_DIR)/algorithms/hungarian.$O \
 $(OBJ_DIR)/algorithms/knapsack_solver.$O \
 $(OBJ_DIR)/algorithms/sparse_permutation.$O

objs/algorithms/dynamic_partition.$O: \
 ortools/algorithms/dynamic_partition.cc \
 ortools/algorithms/dynamic_partition.h ortools/base/logging.h \
 ortools/base/integral_types.h ortools/base/macros.h ortools/base/join.h \
 ortools/base/basictypes.h ortools/base/string_view.h \
 ortools/base/murmur.h ortools/base/thorough_hash.h \
 ortools/base/stringprintf.h | $(OBJ_DIR)/algorithms
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Salgorithms$Sdynamic_partition.cc $(OBJ_OUT)$(OBJ_DIR)$Salgorithms$Sdynamic_partition.$O

objs/algorithms/dynamic_permutation.$O: \
 ortools/algorithms/dynamic_permutation.cc \
 ortools/algorithms/dynamic_permutation.h ortools/base/logging.h \
 ortools/base/integral_types.h ortools/base/macros.h \
 ortools/algorithms/sparse_permutation.h | $(OBJ_DIR)/algorithms
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Salgorithms$Sdynamic_permutation.cc $(OBJ_OUT)$(OBJ_DIR)$Salgorithms$Sdynamic_permutation.$O

 | $(OBJ_DIR)/algorithms
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Salgorithms$Sfind_graph_symmetries.cc $(OBJ_OUT)$(OBJ_DIR)$Salgorithms$Sfind_graph_symmetries.$O

 | $(OBJ_DIR)/algorithms
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Salgorithms$Shungarian.cc $(OBJ_OUT)$(OBJ_DIR)$Salgorithms$Shungarian.$O

 | $(OBJ_DIR)/algorithms
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Salgorithms$Sknapsack_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Salgorithms$Sknapsack_solver.$O

objs/algorithms/sparse_permutation.$O: \
 ortools/algorithms/sparse_permutation.cc \
 ortools/algorithms/sparse_permutation.h ortools/base/logging.h \
 ortools/base/integral_types.h ortools/base/macros.h ortools/base/join.h \
 ortools/base/basictypes.h ortools/base/string_view.h | $(OBJ_DIR)/algorithms
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Salgorithms$Ssparse_permutation.cc $(OBJ_OUT)$(OBJ_DIR)$Salgorithms$Ssparse_permutation.$O

SAT_DEPS = \
 $(SRC_DIR)/ortools/sat/all_different.h \
 $(SRC_DIR)/ortools/sat/boolean_problem.h \
 $(SRC_DIR)/ortools/sat/circuit.h \
 $(SRC_DIR)/ortools/sat/clause.h \
 $(SRC_DIR)/ortools/sat/cp_constraints.h \
 $(SRC_DIR)/ortools/sat/cp_model_checker.h \
 $(SRC_DIR)/ortools/sat/cp_model_expand.h \
 $(SRC_DIR)/ortools/sat/cp_model_lns.h \
 $(SRC_DIR)/ortools/sat/cp_model_objective.h \
 $(SRC_DIR)/ortools/sat/cp_model_presolve.h \
 $(SRC_DIR)/ortools/sat/cp_model_search.h \
 $(SRC_DIR)/ortools/sat/cp_model_solver.h \
 $(SRC_DIR)/ortools/sat/cp_model_symmetries.h \
 $(SRC_DIR)/ortools/sat/cp_model_utils.h \
 $(SRC_DIR)/ortools/sat/cumulative.h \
 $(SRC_DIR)/ortools/sat/disjunctive.h \
 $(SRC_DIR)/ortools/sat/drat_checker.h \
 $(SRC_DIR)/ortools/sat/drat_proof_handler.h \
 $(SRC_DIR)/ortools/sat/drat_writer.h \
 $(SRC_DIR)/ortools/sat/encoding.h \
 $(SRC_DIR)/ortools/sat/integer_expr.h \
 $(SRC_DIR)/ortools/sat/integer.h \
 $(SRC_DIR)/ortools/sat/integer_search.h \
 $(SRC_DIR)/ortools/sat/intervals.h \
 $(SRC_DIR)/ortools/sat/linear_programming_constraint.h \
 $(SRC_DIR)/ortools/sat/linear_relaxation.h \
 $(SRC_DIR)/ortools/sat/lns.h \
 $(SRC_DIR)/ortools/sat/lp_utils.h \
 $(SRC_DIR)/ortools/sat/model.h \
 $(SRC_DIR)/ortools/sat/optimization.h \
 $(SRC_DIR)/ortools/sat/overload_checker.h \
 $(SRC_DIR)/ortools/sat/pb_constraint.h \
 $(SRC_DIR)/ortools/sat/precedences.h \
 $(SRC_DIR)/ortools/sat/restart.h \
 $(SRC_DIR)/ortools/sat/sat_base.h \
 $(SRC_DIR)/ortools/sat/sat_decision.h \
 $(SRC_DIR)/ortools/sat/sat_solver.h \
 $(SRC_DIR)/ortools/sat/simplification.h \
 $(SRC_DIR)/ortools/sat/swig_helper.h \
 $(SRC_DIR)/ortools/sat/symmetry.h \
 $(SRC_DIR)/ortools/sat/table.h \
 $(SRC_DIR)/ortools/sat/theta_tree.h \
 $(SRC_DIR)/ortools/sat/timetable_edgefinding.h \
 $(SRC_DIR)/ortools/sat/timetable.h \
 $(SRC_DIR)/ortools/sat/util.h \
 $(GEN_DIR)/ortools/sat/boolean_problem.pb.h \
 $(GEN_DIR)/ortools/sat/cp_model.pb.h \
 $(GEN_DIR)/ortools/sat/sat_parameters.pb.h

SAT_LIB_OBJS = \
 $(OBJ_DIR)/sat/all_different.$O \
 $(OBJ_DIR)/sat/boolean_problem.$O \
 $(OBJ_DIR)/sat/circuit.$O \
 $(OBJ_DIR)/sat/clause.$O \
 $(OBJ_DIR)/sat/cp_constraints.$O \
 $(OBJ_DIR)/sat/cp_model_checker.$O \
 $(OBJ_DIR)/sat/cp_model_expand.$O \
 $(OBJ_DIR)/sat/cp_model_lns.$O \
 $(OBJ_DIR)/sat/cp_model_objective.$O \
 $(OBJ_DIR)/sat/cp_model_presolve.$O \
 $(OBJ_DIR)/sat/cp_model_search.$O \
 $(OBJ_DIR)/sat/cp_model_solver.$O \
 $(OBJ_DIR)/sat/cp_model_symmetries.$O \
 $(OBJ_DIR)/sat/cp_model_utils.$O \
 $(OBJ_DIR)/sat/cumulative.$O \
 $(OBJ_DIR)/sat/disjunctive.$O \
 $(OBJ_DIR)/sat/drat_checker.$O \
 $(OBJ_DIR)/sat/drat_proof_handler.$O \
 $(OBJ_DIR)/sat/drat_writer.$O \
 $(OBJ_DIR)/sat/encoding.$O \
 $(OBJ_DIR)/sat/integer.$O \
 $(OBJ_DIR)/sat/integer_expr.$O \
 $(OBJ_DIR)/sat/integer_search.$O \
 $(OBJ_DIR)/sat/intervals.$O \
 $(OBJ_DIR)/sat/linear_programming_constraint.$O \
 $(OBJ_DIR)/sat/linear_relaxation.$O \
 $(OBJ_DIR)/sat/lp_utils.$O \
 $(OBJ_DIR)/sat/optimization.$O \
 $(OBJ_DIR)/sat/overload_checker.$O \
 $(OBJ_DIR)/sat/pb_constraint.$O \
 $(OBJ_DIR)/sat/precedences.$O \
 $(OBJ_DIR)/sat/restart.$O \
 $(OBJ_DIR)/sat/sat_decision.$O \
 $(OBJ_DIR)/sat/sat_solver.$O \
 $(OBJ_DIR)/sat/simplification.$O \
 $(OBJ_DIR)/sat/symmetry.$O \
 $(OBJ_DIR)/sat/table.$O \
 $(OBJ_DIR)/sat/theta_tree.$O \
 $(OBJ_DIR)/sat/timetable.$O \
 $(OBJ_DIR)/sat/timetable_edgefinding.$O \
 $(OBJ_DIR)/sat/util.$O \
 $(OBJ_DIR)/sat/boolean_problem.pb.$O \
 $(OBJ_DIR)/sat/cp_model.pb.$O \
 $(OBJ_DIR)/sat/sat_parameters.pb.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sall_different.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sall_different.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sboolean_problem.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sboolean_problem.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scircuit.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scircuit.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sclause.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sclause.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_constraints.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_constraints.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_checker.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_checker.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_expand.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_expand.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_lns.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_lns.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_objective.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_objective.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_presolve.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_presolve.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_search.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_search.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_solver.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_symmetries.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_symmetries.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scp_model_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model_utils.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Scumulative.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scumulative.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sdisjunctive.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sdisjunctive.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sdrat_checker.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sdrat_checker.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sdrat_proof_handler.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sdrat_proof_handler.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sdrat_writer.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sdrat_writer.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sencoding.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sencoding.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sinteger.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sinteger.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sinteger_expr.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sinteger_expr.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sinteger_search.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sinteger_search.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sintervals.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sintervals.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Slinear_programming_constraint.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Slinear_programming_constraint.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Slinear_relaxation.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Slinear_relaxation.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Slp_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Slp_utils.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Soptimization.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Soptimization.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Soverload_checker.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Soverload_checker.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Spb_constraint.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Spb_constraint.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sprecedences.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sprecedences.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Srestart.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Srestart.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Ssat_decision.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Ssat_decision.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Ssat_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Ssat_solver.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Ssimplification.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Ssimplification.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Ssymmetry.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Ssymmetry.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Stable.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Stable.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Stheta_tree.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Stheta_tree.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Stimetable.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Stimetable.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Stimetable_edgefinding.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Stimetable_edgefinding.$O

 | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Ssat$Sutil.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sutil.$O

ortools/sat/boolean_problem.proto: ;

$(GEN_DIR)/ortools/sat/boolean_problem.pb.cc: \
 $(SRC_DIR)/ortools/sat/boolean_problem.proto | $(GEN_DIR)/ortools/sat
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/sat/boolean_problem.proto

$(GEN_DIR)/ortools/sat/boolean_problem.pb.h: \
 $(GEN_DIR)/ortools/sat/boolean_problem.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Ssat$Sboolean_problem.pb.h

$(OBJ_DIR)/sat/boolean_problem.pb.$O: \
 $(GEN_DIR)/ortools/sat/boolean_problem.pb.cc | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Ssat$Sboolean_problem.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Sboolean_problem.pb.$O

ortools/sat/cp_model.proto: ;

$(GEN_DIR)/ortools/sat/cp_model.pb.cc: \
 $(SRC_DIR)/ortools/sat/cp_model.proto | $(GEN_DIR)/ortools/sat
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/sat/cp_model.proto

$(GEN_DIR)/ortools/sat/cp_model.pb.h: \
 $(GEN_DIR)/ortools/sat/cp_model.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Ssat$Scp_model.pb.h

$(OBJ_DIR)/sat/cp_model.pb.$O: \
 $(GEN_DIR)/ortools/sat/cp_model.pb.cc | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Ssat$Scp_model.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Scp_model.pb.$O

ortools/sat/sat_parameters.proto: ;

$(GEN_DIR)/ortools/sat/sat_parameters.pb.cc: \
 $(SRC_DIR)/ortools/sat/sat_parameters.proto | $(GEN_DIR)/ortools/sat
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/sat/sat_parameters.proto

$(GEN_DIR)/ortools/sat/sat_parameters.pb.h: \
 $(GEN_DIR)/ortools/sat/sat_parameters.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Ssat$Ssat_parameters.pb.h

$(OBJ_DIR)/sat/sat_parameters.pb.$O: \
 $(GEN_DIR)/ortools/sat/sat_parameters.pb.cc | $(OBJ_DIR)/sat
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Ssat$Ssat_parameters.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Ssat$Ssat_parameters.pb.$O

BOP_DEPS = \
 $(SRC_DIR)/ortools/bop/bop_base.h \
 $(SRC_DIR)/ortools/bop/bop_fs.h \
 $(SRC_DIR)/ortools/bop/bop_lns.h \
 $(SRC_DIR)/ortools/bop/bop_ls.h \
 $(SRC_DIR)/ortools/bop/bop_portfolio.h \
 $(SRC_DIR)/ortools/bop/bop_solution.h \
 $(SRC_DIR)/ortools/bop/bop_solver.h \
 $(SRC_DIR)/ortools/bop/bop_types.h \
 $(SRC_DIR)/ortools/bop/bop_util.h \
 $(SRC_DIR)/ortools/bop/complete_optimizer.h \
 $(SRC_DIR)/ortools/bop/integral_solver.h \
 $(GEN_DIR)/ortools/bop/bop_parameters.pb.h

BOP_LIB_OBJS = \
 $(OBJ_DIR)/bop/bop_base.$O \
 $(OBJ_DIR)/bop/bop_fs.$O \
 $(OBJ_DIR)/bop/bop_lns.$O \
 $(OBJ_DIR)/bop/bop_ls.$O \
 $(OBJ_DIR)/bop/bop_portfolio.$O \
 $(OBJ_DIR)/bop/bop_solution.$O \
 $(OBJ_DIR)/bop/bop_solver.$O \
 $(OBJ_DIR)/bop/bop_util.$O \
 $(OBJ_DIR)/bop/complete_optimizer.$O \
 $(OBJ_DIR)/bop/integral_solver.$O \
 $(OBJ_DIR)/bop/bop_parameters.pb.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_base.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_base.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_fs.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_fs.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_lns.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_lns.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_ls.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_ls.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_portfolio.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_portfolio.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_solution.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_solution.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_solver.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sbop_util.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_util.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Scomplete_optimizer.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Scomplete_optimizer.$O

 | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sbop$Sintegral_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sintegral_solver.$O

ortools/bop/bop_parameters.proto: ;

$(GEN_DIR)/ortools/bop/bop_parameters.pb.cc: \
 $(SRC_DIR)/ortools/bop/bop_parameters.proto | $(GEN_DIR)/ortools/bop
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/bop/bop_parameters.proto

$(GEN_DIR)/ortools/bop/bop_parameters.pb.h: \
 $(GEN_DIR)/ortools/bop/bop_parameters.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sbop$Sbop_parameters.pb.h

$(OBJ_DIR)/bop/bop_parameters.pb.$O: \
 $(GEN_DIR)/ortools/bop/bop_parameters.pb.cc | $(OBJ_DIR)/bop
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sbop$Sbop_parameters.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sbop$Sbop_parameters.pb.$O

LP_DEPS = \
 $(SRC_DIR)/ortools/linear_solver/glop_utils.h \
 $(SRC_DIR)/ortools/linear_solver/linear_expr.h \
 $(SRC_DIR)/ortools/linear_solver/linear_solver.h \
 $(SRC_DIR)/ortools/linear_solver/model_exporter.h \
 $(SRC_DIR)/ortools/linear_solver/model_validator.h \
 $(GEN_DIR)/ortools/linear_solver/linear_solver.pb.h

LP_LIB_OBJS = \
 $(OBJ_DIR)/linear_solver/bop_interface.$O \
 $(OBJ_DIR)/linear_solver/cbc_interface.$O \
 $(OBJ_DIR)/linear_solver/clp_interface.$O \
 $(OBJ_DIR)/linear_solver/cplex_interface.$O \
 $(OBJ_DIR)/linear_solver/glop_interface.$O \
 $(OBJ_DIR)/linear_solver/glop_utils.$O \
 $(OBJ_DIR)/linear_solver/glpk_interface.$O \
 $(OBJ_DIR)/linear_solver/gurobi_interface.$O \
 $(OBJ_DIR)/linear_solver/linear_expr.$O \
 $(OBJ_DIR)/linear_solver/linear_solver.$O \
 $(OBJ_DIR)/linear_solver/model_exporter.$O \
 $(OBJ_DIR)/linear_solver/model_validator.$O \
 $(OBJ_DIR)/linear_solver/scip_interface.$O \
 $(OBJ_DIR)/linear_solver/linear_solver.pb.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Sbop_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Sbop_interface.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Scbc_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Scbc_interface.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Sclp_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Sclp_interface.$O

objs/linear_solver/cplex_interface.$O: \
 ortools/linear_solver/cplex_interface.cc | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Scplex_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Scplex_interface.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Sglop_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Sglop_interface.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Sglop_utils.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Sglop_utils.$O

objs/linear_solver/glpk_interface.$O: \
 ortools/linear_solver/glpk_interface.cc | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Sglpk_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Sglpk_interface.$O

objs/linear_solver/gurobi_interface.$O: \
 ortools/linear_solver/gurobi_interface.cc | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Sgurobi_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Sgurobi_interface.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Slinear_expr.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Slinear_expr.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Slinear_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Slinear_solver.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Smodel_exporter.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Smodel_exporter.$O

 | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Smodel_validator.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Smodel_validator.$O

objs/linear_solver/scip_interface.$O: \
 ortools/linear_solver/scip_interface.cc | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Slinear_solver$Sscip_interface.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Sscip_interface.$O

ortools/linear_solver/linear_solver.proto: ;

$(GEN_DIR)/ortools/linear_solver/linear_solver.pb.cc: \
 $(SRC_DIR)/ortools/linear_solver/linear_solver.proto \
 $(GEN_DIR)/ortools/util/optional_boolean.pb.cc | $(GEN_DIR)/ortools/linear_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/linear_solver/linear_solver.proto

$(GEN_DIR)/ortools/linear_solver/linear_solver.pb.h: \
 $(GEN_DIR)/ortools/linear_solver/linear_solver.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Slinear_solver$Slinear_solver.pb.h

$(OBJ_DIR)/linear_solver/linear_solver.pb.$O: \
 $(GEN_DIR)/ortools/linear_solver/linear_solver.pb.cc | $(OBJ_DIR)/linear_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Slinear_solver$Slinear_solver.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Slinear_solver$Slinear_solver.pb.$O

CP_DEPS = \
 $(SRC_DIR)/ortools/constraint_solver/constraint_solver.h \
 $(SRC_DIR)/ortools/constraint_solver/constraint_solveri.h \
 $(SRC_DIR)/ortools/constraint_solver/hybrid.h \
 $(SRC_DIR)/ortools/constraint_solver/routing_flags.h \
 $(SRC_DIR)/ortools/constraint_solver/routing.h \
 $(SRC_DIR)/ortools/constraint_solver/routing_neighborhoods.h \
 $(SRC_DIR)/ortools/constraint_solver/routing_types.h \
 $(SRC_DIR)/ortools/constraint_solver/sat_constraint.h \
 $(GEN_DIR)/ortools/constraint_solver/assignment.pb.h \
 $(GEN_DIR)/ortools/constraint_solver/demon_profiler.pb.h \
 $(GEN_DIR)/ortools/constraint_solver/model.pb.h \
 $(GEN_DIR)/ortools/constraint_solver/routing_enums.pb.h \
 $(GEN_DIR)/ortools/constraint_solver/routing_parameters.pb.h \
 $(GEN_DIR)/ortools/constraint_solver/search_limit.pb.h \
 $(GEN_DIR)/ortools/constraint_solver/solver_parameters.pb.h

CP_LIB_OBJS = \
 $(OBJ_DIR)/constraint_solver/ac4_mdd_reset_table.$O \
 $(OBJ_DIR)/constraint_solver/ac4r_table.$O \
 $(OBJ_DIR)/constraint_solver/alldiff_cst.$O \
 $(OBJ_DIR)/constraint_solver/assignment.$O \
 $(OBJ_DIR)/constraint_solver/collect_variables.$O \
 $(OBJ_DIR)/constraint_solver/constraint_solver.$O \
 $(OBJ_DIR)/constraint_solver/constraints.$O \
 $(OBJ_DIR)/constraint_solver/count_cst.$O \
 $(OBJ_DIR)/constraint_solver/default_search.$O \
 $(OBJ_DIR)/constraint_solver/demon_profiler.$O \
 $(OBJ_DIR)/constraint_solver/deviation.$O \
 $(OBJ_DIR)/constraint_solver/diffn.$O \
 $(OBJ_DIR)/constraint_solver/element.$O \
 $(OBJ_DIR)/constraint_solver/expr_array.$O \
 $(OBJ_DIR)/constraint_solver/expr_cst.$O \
 $(OBJ_DIR)/constraint_solver/expressions.$O \
 $(OBJ_DIR)/constraint_solver/gcc.$O \
 $(OBJ_DIR)/constraint_solver/graph_constraints.$O \
 $(OBJ_DIR)/constraint_solver/hybrid.$O \
 $(OBJ_DIR)/constraint_solver/interval.$O \
 $(OBJ_DIR)/constraint_solver/io.$O \
 $(OBJ_DIR)/constraint_solver/local_search.$O \
 $(OBJ_DIR)/constraint_solver/model_cache.$O \
 $(OBJ_DIR)/constraint_solver/nogoods.$O \
 $(OBJ_DIR)/constraint_solver/pack.$O \
 $(OBJ_DIR)/constraint_solver/range_cst.$O \
 $(OBJ_DIR)/constraint_solver/resource.$O \
 $(OBJ_DIR)/constraint_solver/routing.$O \
 $(OBJ_DIR)/constraint_solver/routing_flags.$O \
 $(OBJ_DIR)/constraint_solver/routing_neighborhoods.$O \
 $(OBJ_DIR)/constraint_solver/routing_search.$O \
 $(OBJ_DIR)/constraint_solver/sat_constraint.$O \
 $(OBJ_DIR)/constraint_solver/sched_constraints.$O \
 $(OBJ_DIR)/constraint_solver/sched_expr.$O \
 $(OBJ_DIR)/constraint_solver/sched_search.$O \
 $(OBJ_DIR)/constraint_solver/search.$O \
 $(OBJ_DIR)/constraint_solver/softgcc.$O \
 $(OBJ_DIR)/constraint_solver/table.$O \
 $(OBJ_DIR)/constraint_solver/timetabling.$O \
 $(OBJ_DIR)/constraint_solver/trace.$O \
 $(OBJ_DIR)/constraint_solver/tree_monitor.$O \
 $(OBJ_DIR)/constraint_solver/utilities.$O \
 $(OBJ_DIR)/constraint_solver/visitor.$O \
 $(OBJ_DIR)/constraint_solver/assignment.pb.$O \
 $(OBJ_DIR)/constraint_solver/demon_profiler.pb.$O \
 $(OBJ_DIR)/constraint_solver/model.pb.$O \
 $(OBJ_DIR)/constraint_solver/routing_enums.pb.$O \
 $(OBJ_DIR)/constraint_solver/routing_parameters.pb.$O \
 $(OBJ_DIR)/constraint_solver/search_limit.pb.$O \
 $(OBJ_DIR)/constraint_solver/solver_parameters.pb.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sac4_mdd_reset_table.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sac4_mdd_reset_table.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sac4r_table.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sac4r_table.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Salldiff_cst.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Salldiff_cst.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sassignment.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sassignment.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Scollect_variables.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Scollect_variables.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sconstraint_solver.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sconstraint_solver.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sconstraints.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sconstraints.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Scount_cst.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Scount_cst.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sdefault_search.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sdefault_search.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sdemon_profiler.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sdemon_profiler.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sdeviation.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sdeviation.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sdiffn.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sdiffn.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Selement.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Selement.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sexpr_array.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sexpr_array.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sexpr_cst.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sexpr_cst.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sexpressions.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sexpressions.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sgcc.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sgcc.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sgraph_constraints.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sgraph_constraints.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Shybrid.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Shybrid.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sinterval.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sinterval.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sio.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sio.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Slocal_search.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Slocal_search.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Smodel_cache.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Smodel_cache.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Snogoods.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Snogoods.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Spack.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Spack.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Srange_cst.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Srange_cst.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sresource.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sresource.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Srouting.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Srouting.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Srouting_flags.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Srouting_flags.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Srouting_neighborhoods.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Srouting_neighborhoods.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Srouting_search.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Srouting_search.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Ssat_constraint.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssat_constraint.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Ssched_constraints.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssched_constraints.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Ssched_expr.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssched_expr.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Ssched_search.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssched_search.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Ssearch.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssearch.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Ssoftgcc.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssoftgcc.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Stable.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Stable.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Stimetabling.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Stimetabling.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Strace.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Strace.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Stree_monitor.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Stree_monitor.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Sutilities.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sutilities.$O

 | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(SRC_DIR)$Sortools$Sconstraint_solver$Svisitor.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Svisitor.$O

ortools/constraint_solver/assignment.proto: ;

$(GEN_DIR)/ortools/constraint_solver/assignment.pb.cc: \
 $(SRC_DIR)/ortools/constraint_solver/assignment.proto | $(GEN_DIR)/ortools/constraint_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/constraint_solver/assignment.proto

$(GEN_DIR)/ortools/constraint_solver/assignment.pb.h: \
 $(GEN_DIR)/ortools/constraint_solver/assignment.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sconstraint_solver$Sassignment.pb.h

$(OBJ_DIR)/constraint_solver/assignment.pb.$O: \
 $(GEN_DIR)/ortools/constraint_solver/assignment.pb.cc | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sconstraint_solver$Sassignment.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sassignment.pb.$O

ortools/constraint_solver/demon_profiler.proto: ;

$(GEN_DIR)/ortools/constraint_solver/demon_profiler.pb.cc: \
 $(SRC_DIR)/ortools/constraint_solver/demon_profiler.proto | $(GEN_DIR)/ortools/constraint_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/constraint_solver/demon_profiler.proto

$(GEN_DIR)/ortools/constraint_solver/demon_profiler.pb.h: \
 $(GEN_DIR)/ortools/constraint_solver/demon_profiler.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sconstraint_solver$Sdemon_profiler.pb.h

$(OBJ_DIR)/constraint_solver/demon_profiler.pb.$O: \
 $(GEN_DIR)/ortools/constraint_solver/demon_profiler.pb.cc | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sconstraint_solver$Sdemon_profiler.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Sdemon_profiler.pb.$O

ortools/constraint_solver/model.proto: ;

$(GEN_DIR)/ortools/constraint_solver/model.pb.cc: \
 $(SRC_DIR)/ortools/constraint_solver/model.proto \
 $(GEN_DIR)/ortools/constraint_solver/search_limit.pb.cc | $(GEN_DIR)/ortools/constraint_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/constraint_solver/model.proto

$(GEN_DIR)/ortools/constraint_solver/model.pb.h: \
 $(GEN_DIR)/ortools/constraint_solver/model.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sconstraint_solver$Smodel.pb.h

$(OBJ_DIR)/constraint_solver/model.pb.$O: \
 $(GEN_DIR)/ortools/constraint_solver/model.pb.cc | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sconstraint_solver$Smodel.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Smodel.pb.$O

ortools/constraint_solver/routing_enums.proto: ;

$(GEN_DIR)/ortools/constraint_solver/routing_enums.pb.cc: \
 $(SRC_DIR)/ortools/constraint_solver/routing_enums.proto | $(GEN_DIR)/ortools/constraint_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/constraint_solver/routing_enums.proto

$(GEN_DIR)/ortools/constraint_solver/routing_enums.pb.h: \
 $(GEN_DIR)/ortools/constraint_solver/routing_enums.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sconstraint_solver$Srouting_enums.pb.h

$(OBJ_DIR)/constraint_solver/routing_enums.pb.$O: \
 $(GEN_DIR)/ortools/constraint_solver/routing_enums.pb.cc | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sconstraint_solver$Srouting_enums.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Srouting_enums.pb.$O

ortools/constraint_solver/routing_parameters.proto: ;

$(GEN_DIR)/ortools/constraint_solver/routing_parameters.pb.cc: \
 $(SRC_DIR)/ortools/constraint_solver/routing_parameters.proto \
 $(GEN_DIR)/ortools/constraint_solver/routing_enums.pb.cc \
 $(GEN_DIR)/ortools/constraint_solver/solver_parameters.pb.cc | $(GEN_DIR)/ortools/constraint_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/constraint_solver/routing_parameters.proto

$(GEN_DIR)/ortools/constraint_solver/routing_parameters.pb.h: \
 $(GEN_DIR)/ortools/constraint_solver/routing_parameters.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sconstraint_solver$Srouting_parameters.pb.h

$(OBJ_DIR)/constraint_solver/routing_parameters.pb.$O: \
 $(GEN_DIR)/ortools/constraint_solver/routing_parameters.pb.cc | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sconstraint_solver$Srouting_parameters.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Srouting_parameters.pb.$O

ortools/constraint_solver/search_limit.proto: ;

$(GEN_DIR)/ortools/constraint_solver/search_limit.pb.cc: \
 $(SRC_DIR)/ortools/constraint_solver/search_limit.proto | $(GEN_DIR)/ortools/constraint_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/constraint_solver/search_limit.proto

$(GEN_DIR)/ortools/constraint_solver/search_limit.pb.h: \
 $(GEN_DIR)/ortools/constraint_solver/search_limit.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sconstraint_solver$Ssearch_limit.pb.h

$(OBJ_DIR)/constraint_solver/search_limit.pb.$O: \
 $(GEN_DIR)/ortools/constraint_solver/search_limit.pb.cc | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sconstraint_solver$Ssearch_limit.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssearch_limit.pb.$O

ortools/constraint_solver/solver_parameters.proto: ;

$(GEN_DIR)/ortools/constraint_solver/solver_parameters.pb.cc: \
 $(SRC_DIR)/ortools/constraint_solver/solver_parameters.proto | $(GEN_DIR)/ortools/constraint_solver
	$(PROTOC) --proto_path=$(INC_DIR) $(PROTOBUF_PROTOC_INC) --cpp_out=$(GEN_PATH) $(SRC_DIR)/ortools/constraint_solver/solver_parameters.proto

$(GEN_DIR)/ortools/constraint_solver/solver_parameters.pb.h: \
 $(GEN_DIR)/ortools/constraint_solver/solver_parameters.pb.cc
	$(TOUCH) $(GEN_PATH)$Sortools$Sconstraint_solver$Ssolver_parameters.pb.h

$(OBJ_DIR)/constraint_solver/solver_parameters.pb.$O: \
 $(GEN_DIR)/ortools/constraint_solver/solver_parameters.pb.cc | $(OBJ_DIR)/constraint_solver
	$(CCC) $(CFLAGS) -c $(GEN_PATH)$Sortools$Sconstraint_solver$Ssolver_parameters.pb.cc $(OBJ_OUT)$(OBJ_DIR)$Sconstraint_solver$Ssolver_parameters.pb.$O

