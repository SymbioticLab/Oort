00:41:06,891 INFO tensor([0.6612], device='cuda:0')
00:41:06,898 INFO ===== Experiment start =====
00:41:17,13 INFO ====Initiating DataPartitioner takes 1.525688886642456 s

00:43:28,763 INFO ====Initiating DataPartitioner takes 0.19316720962524414 s

00:43:29,303 INFO ========= Start of Random Partition =========

00:43:30,728 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,728 INFO ========= End of Class/Worker =========

00:43:30,733 INFO ====Data length for client 7 is 1901
00:43:30,971 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=8, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:30,997 INFO ====Data length for client 7 is 83
00:43:32,670 INFO For client 8, upload iter 1, epoch 0, Batch 1/3, Loss:6.3934 | TotalTime 1.647 | Comptime: 1.0269 

00:43:32,699 INFO For client 8, upload iter 1, epoch 0, Batch 2/3, Loss:6.2048 | TotalTime 0.0281 | Comptime: 0.0249 

00:43:32,726 INFO For client 8, upload iter 1, epoch 1, Batch 3/3, Loss:5.5741 | TotalTime 0.0274 | Comptime: 0.0248 

00:43:32,728 INFO ====Data length for client 7 is 83
00:43:33,5 INFO For client 8, upload iter 1, epoch 1, Batch 0/3, Loss:4.9146 | TotalTime 0.2783 | Comptime: 0.0276 

00:43:33,58 INFO For client 8, upload iter 1, epoch 1, Batch 1/3, Loss:4.4996 | TotalTime 0.0526 | Comptime: 0.0247 

00:43:33,86 INFO For client 8, upload iter 1, epoch 2, Batch 2/3, Loss:4.8634 | TotalTime 0.0274 | Comptime: 0.0246 

00:43:33,87 INFO ====Data length for client 7 is 83
00:43:33,398 INFO For client 8, upload iter 1, epoch 2, Batch 3/3, Loss:5.1058 | TotalTime 0.3116 | Comptime: 0.0276 

00:43:33,437 INFO For client 8, upload iter 1, epoch 2, Batch 0/3, Loss:4.7665 | TotalTime 0.0382 | Comptime: 0.0284 

00:43:33,464 INFO For client 8, upload iter 1, epoch 3, Batch 1/3, Loss:4.4342 | TotalTime 0.0269 | Comptime: 0.0246 

00:43:33,465 INFO ====Data length for client 7 is 83
00:43:33,801 INFO For client 8, upload iter 1, epoch 3, Batch 2/3, Loss:4.099 | TotalTime 0.3372 | Comptime: 0.0308 

00:43:33,827 INFO For client 8, upload iter 1, epoch 3, Batch 3/3, Loss:3.8981 | TotalTime 0.0256 | Comptime: 0.0226 

00:43:33,856 INFO For client 8, upload iter 1, epoch 4, Batch 0/3, Loss:4.2331 | TotalTime 0.0281 | Comptime: 0.0255 

00:43:33,857 INFO ====Data length for client 7 is 83
00:43:34,180 INFO For client 8, upload iter 1, epoch 4, Batch 1/3, Loss:3.8873 | TotalTime 0.3241 | Comptime: 0.0299 

00:43:34,209 INFO For client 8, upload iter 1, epoch 4, Batch 2/3, Loss:4.0753 | TotalTime 0.028 | Comptime: 0.0251 

00:43:34,236 INFO For client 8, upload iter 1, epoch 5, Batch 3/3, Loss:3.956 | TotalTime 0.0272 | Comptime: 0.0244 

00:43:34,241 INFO ====Data length for client 7 is 83
00:43:34,561 INFO For client 8, upload iter 1, epoch 5, Batch 0/3, Loss:3.9095 | TotalTime 0.3208 | Comptime: 0.0303 

00:43:34,587 INFO For client 8, upload iter 1, epoch 5, Batch 1/3, Loss:4.1358 | TotalTime 0.0262 | Comptime: 0.023 

00:43:34,617 INFO For client 8, upload iter 1, epoch 6, Batch 2/3, Loss:3.433 | TotalTime 0.029 | Comptime: 0.0251 

00:43:34,618 INFO ====Data length for client 7 is 83
00:43:34,952 INFO For client 8, upload iter 1, epoch 6, Batch 3/3, Loss:3.8878 | TotalTime 0.3344 | Comptime: 0.0317 

00:43:35,33 INFO For client 8, upload iter 1, epoch 6, Batch 0/3, Loss:2.9028 | TotalTime 0.0322 | Comptime: 0.0285 

Terminated
