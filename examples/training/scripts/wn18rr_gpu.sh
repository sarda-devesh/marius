# preprocess the wn18rr graph and put preprocessed graph into output dir
marius_preprocess wn18rr output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/wn18rr_gpu.ini info