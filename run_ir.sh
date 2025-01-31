sleep 1
export CUDA_DEVICE_ORDER="PCI_BUS_ID" 

export CUDA_HOME="/usr/local/cuda-11.7"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.7/bin:$PATH"


total_size=1862
chunk_size=$((total_size / 6))

# Start index for the first chunk
start_index=0

# End index for the first chunk, -1 since end index is inclusive
end_index=$((chunk_size - 1))

for i in {0..5}; do
    gpu_id=$(((i % 3)+1))  # Cycle through GPU IDs: 0, 1, 2
    query_file="query_mmlu_test_open$((i+1)).tsv"
    save_file="mmlu_doc_open_test$((i+1)).json"

    # Run the program with specified GPU and data range
    CUDA_VISIBLE_DEVICES=$gpu_id python pre_ir.py $start_index $end_index $query_file $save_file &

    # Update start and end indices for the next chunk
    start_index=$((end_index + 1))
    end_index=$((start_index + chunk_size - 1))

    # Make sure we don't go past the total size for the last chunk
    if [ $i -eq 5 ]; then
        end_index=$((total_size - 1))
    fi
done
wait  # Wait for all background processes to finish