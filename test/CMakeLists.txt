project(MlLib LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_TAG v1.16.0
        GIT_REPOSITORY https://github.com/google/googletest.git
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(tensor)

target_link_libraries(ml-lib-tests PRIVATE GTest::gtest_main GTest::gmock ml-lib)