00:41:07,510 INFO tensor([0.8551], device='cuda:0')
00:41:07,515 INFO ===== Experiment start =====
00:41:18,395 INFO ====Initiating DataPartitioner takes 1.5085582733154297 s

00:43:28,758 INFO ====Initiating DataPartitioner takes 0.19060754776000977 s

00:43:29,302 INFO ========= Start of Random Partition =========

00:43:30,714 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,714 INFO ========= End of Class/Worker =========

00:43:30,720 INFO ====Data length for client 1 is 1901
00:43:30,992 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=2, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:31,17 INFO ====Data length for client 1 is 18
00:43:32,335 INFO For client 2, upload iter 1, epoch 1, Batch 1/1, Loss:6.4649 | TotalTime 1.2927 | Comptime: 0.6891 

00:43:32,337 INFO ====Data length for client 1 is 18
00:43:32,650 INFO For client 2, upload iter 1, epoch 2, Batch 0/1, Loss:6.1885 | TotalTime 0.3138 | Comptime: 0.0344 

00:43:32,651 INFO ====Data length for client 1 is 18
00:43:32,994 INFO For client 2, upload iter 1, epoch 3, Batch 1/1, Loss:5.2278 | TotalTime 0.3442 | Comptime: 0.0329 

00:43:32,996 INFO ====Data length for client 1 is 18
00:43:33,343 INFO For client 2, upload iter 1, epoch 4, Batch 0/1, Loss:5.4707 | TotalTime 0.3475 | Comptime: 0.0314 

00:43:33,344 INFO ====Data length for client 1 is 18
00:43:33,694 INFO For client 2, upload iter 1, epoch 5, Batch 1/1, Loss:4.7582 | TotalTime 0.3506 | Comptime: 0.0328 

00:43:33,696 INFO ====Data length for client 1 is 18
00:43:34,31 INFO For client 2, upload iter 1, epoch 6, Batch 0/1, Loss:5.0059 | TotalTime 0.3365 | Comptime: 0.0302 

00:43:34,33 INFO ====Data length for client 1 is 18
00:43:34,373 INFO For client 2, upload iter 1, epoch 7, Batch 1/1, Loss:4.7884 | TotalTime 0.3404 | Comptime: 0.0312 

00:43:34,374 INFO ====Data length for client 1 is 18
00:43:34,731 INFO For client 2, upload iter 1, epoch 8, Batch 0/1, Loss:4.6544 | TotalTime 0.3573 | Comptime: 0.0294 

00:43:34,732 INFO ====Data length for client 1 is 18
00:43:35,81 INFO For client 2, upload iter 1, epoch 9, Batch 1/1, Loss:4.4494 | TotalTime 0.3493 | Comptime: 0.0293 

00:43:35,82 INFO ====Data length for client 1 is 18
00:43:35,413 INFO For client 2, upload iter 1, epoch 10, Batch 0/1, Loss:4.2464 | TotalTime 0.3319 | Comptime: 0.0324 

00:43:35,415 INFO ====Data length for client 1 is 18
00:43:35,754 INFO For client 2, upload iter 1, epoch 11, Batch 1/1, Loss:4.3548 | TotalTime 0.3395 | Comptime: 0.0326 

00:43:35,755 INFO ====Data length for client 1 is 18
00:43:36,113 INFO For client 2, upload iter 1, epoch 12, Batch 0/1, Loss:4.1709 | TotalTime 0.359 | Comptime: 0.0325 

00:43:36,118 INFO ====Data length for client 1 is 18
00:43:36,474 INFO For client 2, upload iter 1, epoch 13, Batch 1/1, Loss:3.9779 | TotalTime 0.357 | Comptime: 0.0331 

00:43:36,476 INFO ====Data length for client 1 is 18
00:43:36,810 INFO For client 2, upload iter 1, epoch 14, Batch 0/1, Loss:4.2682 | TotalTime 0.3351 | Comptime: 0.0295 

00:43:36,812 INFO ====Data length for client 1 is 18
00:43:37,151 INFO For client 2, upload iter 1, epoch 15, Batch 1/1, Loss:3.9011 | TotalTime 0.3395 | Comptime: 0.0293 

00:43:37,152 INFO ====Data length for client 1 is 18
00:43:37,501 INFO For client 2, upload iter 1, epoch 16, Batch 0/1, Loss:3.8968 | TotalTime 0.3501 | Comptime: 0.03 

00:43:37,503 INFO ====Data length for client 1 is 18
00:43:37,850 INFO For client 2, upload iter 1, epoch 17, Batch 1/1, Loss:3.8627 | TotalTime 0.3484 | Comptime: 0.0295 

00:43:37,852 INFO ====Data length for client 1 is 18
00:43:38,193 INFO For client 2, upload iter 1, epoch 18, Batch 0/1, Loss:3.7853 | TotalTime 0.3418 | Comptime: 0.0317 

00:43:38,195 INFO ====Data length for client 1 is 18
00:43:38,555 INFO For client 2, upload iter 1, epoch 19, Batch 1/1, Loss:3.8782 | TotalTime 0.3608 | Comptime: 0.0332 

00:43:38,556 INFO ====Data length for client 1 is 18
00:43:38,890 INFO For client 2, upload iter 1, epoch 20, Batch 0/1, Loss:3.724 | TotalTime 0.3347 | Comptime: 0.0292 

Terminated
