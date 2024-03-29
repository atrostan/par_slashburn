cmake_minimum_required(VERSION 3.16)
project(par_slashburn)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

Set(FETCHCONTENT_QUIET FALSE)


FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
        GIT_PROGRESS TRUE
        )

FetchContent_MakeAvailable(fmt)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -pedantic -fopenmp")

find_package(Boost 1.58.0 EXACT COMPONENTS system filesystem graph serialization REQUIRED)
find_package(OpenMP REQUIRED)

set(absl_DIR ./install/lib/cmake/absl)
find_package(oneDPL REQUIRED)
add_subdirectory(ips4o)
add_subdirectory(parallel-hashmap)
find_package(absl REQUIRED)

set(GraphPreprocessDir "../graph_preprocess")

add_executable(par_slashburn main.cpp cc.cpp cc.h graph.h timer.h util.h pvector.h platform_atomics.h bitmap.h benchmark.h builder.h reader.h generator.h command_line.h writer.h utilities.cpp utilities.h par_utils.cpp par_utils.h SlashBurn.cpp SlashBurn.h ${GraphPreprocessDir}/sql.h ${GraphPreprocessDir}/sql.cpp )

add_executable(pr pr.cpp graph.h timer.h util.h pvector.h platform_atomics.h bitmap.h benchmark.h builder.h reader.h generator.h command_line.h writer.h)
add_executable(compute_ccs compute_ccs.cpp graph.h timer.h util.h pvector.h platform_atomics.h bitmap.h benchmark.h builder.h reader.h generator.h command_line.h writer.h cc.h cc.cpp)

target_include_directories(par_slashburn PUBLIC ${GraphPreprocessDir})

target_link_libraries(par_slashburn PUBLIC ${Boost_LIBRARIES} numa event_pthreads event_core pthread OpenMP::OpenMP_CXX ${FOLLY_LIBRARIES} fmt::fmt ips4o oneDPL absl::flat_hash_set absl::flat_hash_map absl::btree sqlite3)
target_link_libraries(pr PUBLIC ${Boost_LIBRARIES} OpenMP::OpenMP_CXX fmt::fmt oneDPL)
target_link_libraries(compute_ccs PUBLIC ${Boost_LIBRARIES}  fmt::fmt oneDPL)
