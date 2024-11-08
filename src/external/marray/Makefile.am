ACLOCAL_AMFLAGS = -I m4

AM_LDFLAGS = -lpthread
noinst_PROGRAMS = bin/test

if coverage

headers = src/marray/detail/utility.hpp \
          src/marray/detail/vector_avx.hpp \
          src/marray/detail/vector_avx512.hpp \
          src/marray/detail/vector_neon.hpp \
          src/marray/detail/vector_sse41.hpp \
          src/marray/detail/vector.hpp \
          src/marray/dpd/dpd_marray_base.hpp \
          src/marray/dpd/dpd_marray_view.hpp \
          src/marray/dpd/dpd_marray.hpp \
          src/marray/dpd/dpd_range.hpp \
          src/marray/fwd/expression_fwd.hpp \
          src/marray/fwd/marray_fwd.hpp \
          src/marray/indexed/indexed_marray_base.hpp \
          src/marray/indexed/indexed_marray_view.hpp \
          src/marray/indexed/indexed_marray.hpp \
          src/marray/indexed_dpd/indexed_dpd_marray_base.hpp \
          src/marray/indexed_dpd/indexed_dpd_marray_view.hpp \
          src/marray/indexed_dpd/indexed_dpd_marray.hpp \
          src/marray/array_1d.hpp \
          src/marray/array_2d.hpp \
          src/marray/expression.hpp \
          src/marray/index_iterator.hpp \
          src/marray/marray_base.hpp \
          src/marray/marray_iterator.hpp \
          src/marray/marray_slice.hpp \
          src/marray/marray_view.hpp \
          src/marray/marray.hpp \
          src/marray/range.hpp \
          src/marray/rotate.hpp \
          src/marray/short_vector.hpp \
          src/marray/types.hpp \
          src/third_party/catch2/catch_amalgamated.hpp

bin_test_SOURCES = src/test/dpd_marray_view.cxx \
                   src/test/dpd_marray.cxx \
                   src/test/dpd_range.cxx \
                   src/test/expression.cxx \
                   src/test/index_iterator.cxx \
                   src/test/indexed_dpd_marray.cxx \
                   src/test/indexed_marray.cxx \
                   src/test/marray_view.cxx \
                   src/test/marray.cxx \
                   src/test/range.cxx \
                   src/test/short_vector.cxx \
                   src/third_party/catch2/catch_amalgamated.cpp

AM_CPPFLAGS = -DMARRAY_ENABLE_ASSERTS -Isrc/marray -Isrc/third_party/catch2
AM_CXXFLAGS = -fprofile-instr-generate -fcoverage-mapping
VPATH += ./src
FORCE_COVER_DIR = force-cover-3.0
FORCE_COVER = $(FORCE_COVER_DIR)/force_cover
EXTRA_bin_test_DEPENDENCIES = $(headers)

src/marray/%.hpp: $(srcdir)/marray/%.hpp $(FORCE_COVER)
	@mkdir -p $(dir $@)
	$(FORCE_COVER) $< -- -xc++ -std=c++17 -I$(srcdir)/marray -I$(srcdir)/third_party/catch2 \
	    -DMARRAY_ENABLE_ASSERTS -resource-dir `clang++ -print-resource-dir` > $@

src/test/%.cxx: $(srcdir)/test/%.cxx $(FORCE_COVER) $(headers)
	@mkdir -p $(dir $@)
	$(FORCE_COVER) $< -- -xc++ -std=c++17 -I$(srcdir)/marray -I$(srcdir)/third_party/catch2 \
	    -DMARRAY_ENABLE_ASSERTS -resource-dir `clang++ -print-resource-dir` > $@

src/third_party/catch2/catch_amalgamated.o: src/third_party/catch2/catch_amalgamated.hpp

src/third_party/catch2/%: $(srcdir)/third_party/catch2/%
	@mkdir -p $(dir $@)
	@cp $< $@

$(FORCE_COVER): $(FORCE_COVER_DIR)
	clang++ -O0 -g -std=c++14 `llvm-config --cxxflags` $(FORCE_COVER_DIR)/force_cover.cpp -o $@ \
	    -lclangTooling -lclangASTMatchers -lclangFormat -lclangFrontend -lclangDriver -lclangParse \
	    -lclangSerialization -lclangSema -lclangEdit -lclangAnalysis -lclangToolingCore -lclangAST -lclangRewrite \
	    -lclangLex -lclangBasic -lclangFrontendTool -lclangRewriteFrontend -lclangStaticAnalyzerFrontend \
	    -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangAPINotes \
	    `llvm-config --ldflags --libs --system-libs`

$(FORCE_COVER_DIR):
	wget https://github.com/emilydolson/force-cover/archive/refs/tags/v3.0.tar.gz
	tar zxvf v3.0.tar.gz

else

bin_test_SOURCES = test/dpd_marray_view.cxx \
                   test/dpd_marray.cxx \
                   test/dpd_range.cxx \
                   test/expression.cxx \
                   test/index_iterator.cxx \
                   test/indexed_dpd_marray.cxx \
                   test/indexed_marray.cxx \
                   test/marray_view.cxx \
                   test/marray.cxx \
                   test/range.cxx \
                   test/short_vector.cxx \
                   third_party/catch2/catch_amalgamated.cpp

AM_CPPFLAGS = -DMARRAY_ENABLE_ASSERTS -I$(srcdir)/marray -I$(srcdir)/third_party/catch2
VPATH += $(srcdir)

endif

TESTS = bin/test
