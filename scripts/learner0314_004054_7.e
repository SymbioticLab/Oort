00:41:07,577 INFO tensor([0.9533], device='cuda:0')
00:41:07,583 INFO ===== Experiment start =====
00:41:18,558 INFO ====Initiating DataPartitioner takes 1.4972710609436035 s

00:43:28,761 INFO ====Initiating DataPartitioner takes 0.19058823585510254 s

00:43:29,307 INFO ========= Start of Random Partition =========

00:43:30,707 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,708 INFO ========= End of Class/Worker =========

00:43:30,713 INFO ====Data length for client 6 is 1901
00:43:30,949 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=7, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:30,974 INFO ====Data length for client 6 is 4
00:43:31,974 INFO For client 7, upload iter 1, epoch 1, Batch 1/1, Loss:6.4349 | TotalTime 0.9632 | Comptime: 0.7131 

00:43:31,976 INFO ====Data length for client 6 is 4
00:43:32,215 INFO For client 7, upload iter 1, epoch 2, Batch 0/1, Loss:5.5737 | TotalTime 0.24 | Comptime: 0.0348 

00:43:32,216 INFO ====Data length for client 6 is 4
00:43:32,466 INFO For client 7, upload iter 1, epoch 3, Batch 1/1, Loss:2.6922 | TotalTime 0.2502 | Comptime: 0.0277 

00:43:32,467 INFO ====Data length for client 6 is 4
00:43:32,719 INFO For client 7, upload iter 1, epoch 4, Batch 0/1, Loss:3.1441 | TotalTime 0.2527 | Comptime: 0.0313 

00:43:32,720 INFO ====Data length for client 6 is 4
00:43:32,978 INFO For client 7, upload iter 1, epoch 5, Batch 1/1, Loss:10.0725 | TotalTime 0.2579 | Comptime: 0.0299 

00:43:32,979 INFO ====Data length for client 6 is 4
00:43:33,237 INFO For client 7, upload iter 1, epoch 6, Batch 0/1, Loss:4.6071 | TotalTime 0.2591 | Comptime: 0.0301 

00:43:33,239 INFO ====Data length for client 6 is 4
00:43:33,489 INFO For client 7, upload iter 1, epoch 7, Batch 1/1, Loss:4.5633 | TotalTime 0.2508 | Comptime: 0.0311 

00:43:33,490 INFO ====Data length for client 6 is 4
00:43:33,738 INFO For client 7, upload iter 1, epoch 8, Batch 0/1, Loss:3.4476 | TotalTime 0.2483 | Comptime: 0.0307 

00:43:33,739 INFO ====Data length for client 6 is 4
00:43:33,998 INFO For client 7, upload iter 1, epoch 9, Batch 1/1, Loss:4.0917 | TotalTime 0.2597 | Comptime: 0.0306 

00:43:34,0 INFO ====Data length for client 6 is 4
00:43:34,258 INFO For client 7, upload iter 1, epoch 10, Batch 0/1, Loss:4.0097 | TotalTime 0.2592 | Comptime: 0.0309 

00:43:34,260 INFO ====Data length for client 6 is 4
00:43:34,519 INFO For client 7, upload iter 1, epoch 11, Batch 1/1, Loss:3.5689 | TotalTime 0.26 | Comptime: 0.0312 

00:43:34,520 INFO ====Data length for client 6 is 4
00:43:34,768 INFO For client 7, upload iter 1, epoch 12, Batch 0/1, Loss:4.2257 | TotalTime 0.248 | Comptime: 0.0303 

00:43:34,772 INFO ====Data length for client 6 is 4
00:43:35,18 INFO For client 7, upload iter 1, epoch 13, Batch 1/1, Loss:3.0952 | TotalTime 0.247 | Comptime: 0.0305 

00:43:35,20 INFO ====Data length for client 6 is 4
00:43:35,268 INFO For client 7, upload iter 1, epoch 14, Batch 0/1, Loss:3.334 | TotalTime 0.2495 | Comptime: 0.0307 

00:43:35,270 INFO ====Data length for client 6 is 4
00:43:35,529 INFO For client 7, upload iter 1, epoch 15, Batch 1/1, Loss:3.2513 | TotalTime 0.2597 | Comptime: 0.0311 

00:43:35,530 INFO ====Data length for client 6 is 4
00:43:35,789 INFO For client 7, upload iter 1, epoch 16, Batch 0/1, Loss:3.0485 | TotalTime 0.2594 | Comptime: 0.0311 

00:43:35,790 INFO ====Data length for client 6 is 4
00:43:36,38 INFO For client 7, upload iter 1, epoch 17, Batch 1/1, Loss:2.8938 | TotalTime 0.2484 | Comptime: 0.0307 

00:43:36,39 INFO ====Data length for client 6 is 4
00:43:36,299 INFO For client 7, upload iter 1, epoch 18, Batch 0/1, Loss:2.6771 | TotalTime 0.26 | Comptime: 0.031 

00:43:36,300 INFO ====Data length for client 6 is 4
00:43:36,558 INFO For client 7, upload iter 1, epoch 19, Batch 1/1, Loss:2.6884 | TotalTime 0.2587 | Comptime: 0.0308 

00:43:36,560 INFO ====Data length for client 6 is 4
00:43:36,819 INFO For client 7, upload iter 1, epoch 20, Batch 0/1, Loss:2.8278 | TotalTime 0.2598 | Comptime: 0.031 

Terminated
