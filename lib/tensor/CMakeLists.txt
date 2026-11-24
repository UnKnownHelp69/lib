target_sources(ml-lib
        PUBLIC
            FILE_SET CXX_MODULES
            FILES
                tensor.cppm
                impls/cpu-tensor.cppm
)