00:41:07,500 INFO tensor([0.3501], device='cuda:0')
00:41:07,516 INFO ===== Experiment start =====
00:41:18,517 INFO ====Initiating DataPartitioner takes 1.5219275951385498 s

00:43:28,758 INFO ====Initiating DataPartitioner takes 0.18950533866882324 s

00:43:29,298 INFO ========= Start of Random Partition =========

00:43:30,712 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,712 INFO ========= End of Class/Worker =========

00:43:30,718 INFO ====Data length for client 8 is 1901
00:43:30,998 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=9, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:31,22 INFO ====Data length for client 8 is 2
00:43:32,512 INFO For client 9, upload iter 1, epoch 1, Batch 1/1, Loss:6.2282 | TotalTime 1.4647 | Comptime: 1.2421 

00:43:32,515 INFO ====Data length for client 8 is 2
00:43:32,731 INFO For client 9, upload iter 1, epoch 2, Batch 0/1, Loss:4.7378 | TotalTime 0.2167 | Comptime: 0.0274 

00:43:32,732 INFO ====Data length for client 8 is 2
00:43:32,981 INFO For client 9, upload iter 1, epoch 3, Batch 1/1, Loss:1.6095 | TotalTime 0.2488 | Comptime: 0.0312 

00:43:32,982 INFO ====Data length for client 8 is 2
00:43:33,220 INFO For client 9, upload iter 1, epoch 4, Batch 0/1, Loss:10.3723 | TotalTime 0.239 | Comptime: 0.0311 

00:43:33,222 INFO ====Data length for client 8 is 2
00:43:33,468 INFO For client 9, upload iter 1, epoch 5, Batch 1/1, Loss:4.5761 | TotalTime 0.2473 | Comptime: 0.0287 

00:43:33,470 INFO ====Data length for client 8 is 2
00:43:33,718 INFO For client 9, upload iter 1, epoch 6, Batch 0/1, Loss:4.3381 | TotalTime 0.2492 | Comptime: 0.0287 

00:43:33,720 INFO ====Data length for client 8 is 2
00:43:33,981 INFO For client 9, upload iter 1, epoch 7, Batch 1/1, Loss:1.8654 | TotalTime 0.262 | Comptime: 0.0313 

00:43:33,982 INFO ====Data length for client 8 is 2
00:43:34,231 INFO For client 9, upload iter 1, epoch 8, Batch 0/1, Loss:3.6224 | TotalTime 0.2496 | Comptime: 0.0317 

00:43:34,233 INFO ====Data length for client 8 is 2
00:43:34,488 INFO For client 9, upload iter 1, epoch 9, Batch 1/1, Loss:1.9512 | TotalTime 0.2559 | Comptime: 0.0284 

00:43:34,489 INFO ====Data length for client 8 is 2
00:43:34,738 INFO For client 9, upload iter 1, epoch 10, Batch 0/1, Loss:0.9488 | TotalTime 0.2495 | Comptime: 0.029 

00:43:34,740 INFO ====Data length for client 8 is 2
00:43:34,978 INFO For client 9, upload iter 1, epoch 11, Batch 1/1, Loss:4.4713 | TotalTime 0.2391 | Comptime: 0.0285 

00:43:34,979 INFO ====Data length for client 8 is 2
00:43:35,229 INFO For client 9, upload iter 1, epoch 12, Batch 0/1, Loss:2.598 | TotalTime 0.2505 | Comptime: 0.0296 

00:43:35,235 INFO ====Data length for client 8 is 2
00:43:35,490 INFO For client 9, upload iter 1, epoch 13, Batch 1/1, Loss:1.7426 | TotalTime 0.2553 | Comptime: 0.0299 

00:43:35,491 INFO ====Data length for client 8 is 2
00:43:35,727 INFO For client 9, upload iter 1, epoch 14, Batch 0/1, Loss:1.9412 | TotalTime 0.2371 | Comptime: 0.0281 

00:43:35,729 INFO ====Data length for client 8 is 2
00:43:35,990 INFO For client 9, upload iter 1, epoch 15, Batch 1/1, Loss:2.1852 | TotalTime 0.2615 | Comptime: 0.0299 

00:43:35,991 INFO ====Data length for client 8 is 2
00:43:36,228 INFO For client 9, upload iter 1, epoch 16, Batch 0/1, Loss:1.0423 | TotalTime 0.2379 | Comptime: 0.0289 

00:43:36,230 INFO ====Data length for client 8 is 2
00:43:36,489 INFO For client 9, upload iter 1, epoch 17, Batch 1/1, Loss:0.8518 | TotalTime 0.2605 | Comptime: 0.0297 

00:43:36,491 INFO ====Data length for client 8 is 2
00:43:36,728 INFO For client 9, upload iter 1, epoch 18, Batch 0/1, Loss:0.7634 | TotalTime 0.2384 | Comptime: 0.0288 

00:43:36,730 INFO ====Data length for client 8 is 2
00:43:36,991 INFO For client 9, upload iter 1, epoch 19, Batch 1/1, Loss:1.2557 | TotalTime 0.2617 | Comptime: 0.0311 

00:43:36,992 INFO ====Data length for client 8 is 2
00:43:37,250 INFO For client 9, upload iter 1, epoch 20, Batch 0/1, Loss:1.5371 | TotalTime 0.2581 | Comptime: 0.03 

Terminated
