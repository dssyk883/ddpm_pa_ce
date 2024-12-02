python ddpm_test.py \
    --test_path ./dataset/test/SNR_test_set \
    --test_noisy_path ./dataset/test_noisy/SNR_test_set \
    --sample_path ./dataset/test_noisy/SNR_test_set/sample \
    --checkpoint ./checkpoints/best_model.pth \
    --batch_size 64 \
    --device cuda:0 \
    --output_dir ./test_results