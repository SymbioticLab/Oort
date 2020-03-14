00:41:07,498 INFO tensor([0.7269], device='cuda:0')
00:41:07,503 INFO ===== Experiment start =====
00:41:18,530 INFO ====Initiating DataPartitioner takes 1.551971197128296 s

00:43:28,761 INFO ====Initiating DataPartitioner takes 0.19470953941345215 s

00:43:29,302 INFO ========= Start of Random Partition =========

00:43:30,703 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,703 INFO ========= End of Class/Worker =========

00:43:30,709 INFO ====Data length for client 2 is 1901
00:43:31,3 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=3, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:31,37 INFO ====Data length for client 2 is 3451
00:43:32,410 INFO For client 3, upload iter 1, epoch 0, Batch 1/108, Loss:6.3997 | TotalTime 1.3471 | Comptime: 1.0521 

00:43:32,436 INFO For client 3, upload iter 1, epoch 0, Batch 2/108, Loss:6.2118 | TotalTime 0.025 | Comptime: 0.0224 

00:43:32,463 INFO For client 3, upload iter 1, epoch 0, Batch 3/108, Loss:5.3448 | TotalTime 0.0269 | Comptime: 0.0237 

00:43:32,489 INFO For client 3, upload iter 1, epoch 0, Batch 4/108, Loss:5.5927 | TotalTime 0.0258 | Comptime: 0.0229 

00:43:32,513 INFO For client 3, upload iter 1, epoch 0, Batch 5/108, Loss:5.1358 | TotalTime 0.0233 | Comptime: 0.0205 

00:43:32,538 INFO For client 3, upload iter 1, epoch 0, Batch 6/108, Loss:5.3551 | TotalTime 0.0247 | Comptime: 0.0218 

00:43:32,562 INFO For client 3, upload iter 1, epoch 0, Batch 7/108, Loss:4.8701 | TotalTime 0.024 | Comptime: 0.0207 

00:43:32,593 INFO For client 3, upload iter 1, epoch 0, Batch 8/108, Loss:5.3306 | TotalTime 0.0303 | Comptime: 0.0275 

00:43:32,621 INFO For client 3, upload iter 1, epoch 0, Batch 9/108, Loss:4.3192 | TotalTime 0.0281 | Comptime: 0.0257 

00:43:32,648 INFO For client 3, upload iter 1, epoch 0, Batch 10/108, Loss:4.7051 | TotalTime 0.0271 | Comptime: 0.0245 

00:43:32,674 INFO For client 3, upload iter 1, epoch 0, Batch 11/108, Loss:4.5373 | TotalTime 0.0249 | Comptime: 0.0219 

00:43:32,710 INFO For client 3, upload iter 1, epoch 0, Batch 12/108, Loss:4.6181 | TotalTime 0.0362 | Comptime: 0.0241 

00:43:32,757 INFO For client 3, upload iter 1, epoch 0, Batch 13/108, Loss:4.2686 | TotalTime 0.0469 | Comptime: 0.0245 

00:43:32,803 INFO For client 3, upload iter 1, epoch 0, Batch 14/108, Loss:4.5471 | TotalTime 0.0458 | Comptime: 0.0241 

00:43:32,863 INFO For client 3, upload iter 1, epoch 0, Batch 15/108, Loss:4.5958 | TotalTime 0.0596 | Comptime: 0.0226 

00:43:32,901 INFO For client 3, upload iter 1, epoch 0, Batch 16/108, Loss:4.6396 | TotalTime 0.038 | Comptime: 0.0229 

00:43:33,3 INFO For client 3, upload iter 1, epoch 0, Batch 17/108, Loss:4.2116 | TotalTime 0.101 | Comptime: 0.0288 

00:43:33,33 INFO For client 3, upload iter 1, epoch 0, Batch 18/108, Loss:3.3371 | TotalTime 0.0267 | Comptime: 0.0242 

00:43:33,92 INFO For client 3, upload iter 1, epoch 0, Batch 19/108, Loss:4.9678 | TotalTime 0.0594 | Comptime: 0.0277 

00:43:33,158 INFO For client 3, upload iter 1, epoch 0, Batch 20/108, Loss:3.9422 | TotalTime 0.0652 | Comptime: 0.0255 

Terminated
