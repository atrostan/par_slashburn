cmake_minimum_required(VERSION 3.24)
project(gapbs_sb)
set(CMAKE_CXX_STANDARD 17)

set(FB_CODE_BUILDER_ROOT_INSTALLED_DIR /home/atrostan/Workspace/folly)
set(TMP_FMT_DIR fmt-ApGa4gLhMboMn3lnlNpO24GHY_vMTOb_1vHzX4cbY_I)
set(TMP_BOOST_DIR boost-IS4whrH121lLAOwBiAUY2LL-OZUxK_SHM8-koWiaFWY)

set(Boost_DIR ${FB_CODE_BUILDER_ROOT_INSTALLED_DIR}/boost-v9EA-dspervZncp6mt8cWn3D0bo8kh0T0ar5pVrWZLA/lib/cmake/Boost-1.78.0)
set(fmt_DIR ${FB_CODE_BUILDER_ROOT_INSTALLED_DIR}/${TMP_FMT_DIR}/lib/cmake/fmt)
set(folly_DIR ${FB_CODE_BUILDER_ROOT_INSTALLED_DIR}/folly/lib/cmake/folly)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -pedantic -fopenmp")

find_package(Boost 1.74.0 COMPONENTS system filesystem graph serialization REQUIRED)
find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(folly REQUIRED)
find_package(gflags REQUIRED)

set(absl_DIR /home/atrostan/CLionProjects/gapbs_sb/install/lib/cmake/absl)
find_package(oneDPL REQUIRED)
add_subdirectory(ips4o)
add_subdirectory(parallel-hashmap)
find_package(absl REQUIRED)

set(GraphPreprocessDir "/home/atrostan/Workspace/repos/congenial-enigma/graph_preprocess")

add_executable(gapbs_sb main.cpp cc.cpp cc.h graph.h timer.h util.h pvector.h platform_atomics.h bitmap.h benchmark.h builder.h reader.h generator.h command_line.h writer.h utilities.cpp utilities.h par_utils.cpp par_utils.h SlashBurn.cpp SlashBurn.h ${GraphPreprocessDir}/sql.h ${GraphPreprocessDir}/sql.cpp)
target_include_directories(gapbs_sb PUBLIC ${GraphPreprocessDir})

target_link_libraries(gapbs_sb PUBLIC ${Boost_LIBRARIES} numa event_pthreads event_core pthread OpenMP::OpenMP_CXX ${FOLLY_LIBRARIES} fmt::fmt ips4o oneDPL absl::flat_hash_set absl::flat_hash_map absl::btree sqlite3)