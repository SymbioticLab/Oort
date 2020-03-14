00:41:06,880 INFO tensor([0.5098], device='cuda:0')
00:41:06,885 INFO ===== Experiment start =====
00:41:16,866 INFO ====Initiating DataPartitioner takes 1.5280735492706299 s

00:43:28,758 INFO ====Initiating DataPartitioner takes 0.18955612182617188 s

00:43:29,299 INFO ========= Start of Random Partition =========

00:43:30,722 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,722 INFO ========= End of Class/Worker =========

00:43:30,728 INFO ====Data length for client 0 is 1901
00:43:30,979 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=1, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:31,5 INFO ====Data length for client 0 is 527
00:43:32,202 INFO For client 1, upload iter 1, epoch 0, Batch 1/17, Loss:6.065 | TotalTime 1.1707 | Comptime: 0.6927 

00:43:32,233 INFO For client 1, upload iter 1, epoch 0, Batch 2/17, Loss:3.544 | TotalTime 0.0304 | Comptime: 0.0275 

00:43:32,259 INFO For client 1, upload iter 1, epoch 0, Batch 3/17, Loss:3.8832 | TotalTime 0.0265 | Comptime: 0.0241 

00:43:32,282 INFO For client 1, upload iter 1, epoch 0, Batch 4/17, Loss:3.8142 | TotalTime 0.0223 | Comptime: 0.0195 

00:43:32,310 INFO For client 1, upload iter 1, epoch 0, Batch 5/17, Loss:4.3414 | TotalTime 0.028 | Comptime: 0.0253 

00:43:32,335 INFO For client 1, upload iter 1, epoch 0, Batch 6/17, Loss:3.2952 | TotalTime 0.0248 | Comptime: 0.0221 

00:43:32,363 INFO For client 1, upload iter 1, epoch 0, Batch 7/17, Loss:4.9754 | TotalTime 0.0268 | Comptime: 0.0238 

00:43:32,503 INFO For client 1, upload iter 1, epoch 0, Batch 8/17, Loss:3.2465 | TotalTime 0.1398 | Comptime: 0.0282 

00:43:32,614 INFO For client 1, upload iter 1, epoch 0, Batch 9/17, Loss:3.1859 | TotalTime 0.1112 | Comptime: 0.0251 

00:43:32,754 INFO For client 1, upload iter 1, epoch 0, Batch 10/17, Loss:2.292 | TotalTime 0.1398 | Comptime: 0.0254 

00:43:32,904 INFO For client 1, upload iter 1, epoch 0, Batch 11/17, Loss:2.5212 | TotalTime 0.1495 | Comptime: 0.0254 

00:43:33,63 INFO For client 1, upload iter 1, epoch 0, Batch 12/17, Loss:2.6808 | TotalTime 0.159 | Comptime: 0.0257 

00:43:33,163 INFO For client 1, upload iter 1, epoch 0, Batch 13/17, Loss:2.7466 | TotalTime 0.0999 | Comptime: 0.0282 

00:43:33,313 INFO For client 1, upload iter 1, epoch 0, Batch 14/17, Loss:2.3844 | TotalTime 0.15 | Comptime: 0.0255 

00:43:33,403 INFO For client 1, upload iter 1, epoch 0, Batch 15/17, Loss:2.8789 | TotalTime 0.0889 | Comptime: 0.0286 

00:43:33,513 INFO For client 1, upload iter 1, epoch 0, Batch 16/17, Loss:3.4067 | TotalTime 0.1101 | Comptime: 0.0258 

00:43:33,604 INFO For client 1, upload iter 1, epoch 1, Batch 17/17, Loss:3.3356 | TotalTime 0.0909 | Comptime: 0.0248 

00:43:33,613 INFO ====Data length for client 0 is 527
00:43:34,21 INFO For client 1, upload iter 1, epoch 1, Batch 0/17, Loss:1.1586 | TotalTime 0.4138 | Comptime: 0.0335 

00:43:34,103 INFO For client 1, upload iter 1, epoch 1, Batch 1/17, Loss:4.4321 | TotalTime 0.0709 | Comptime: 0.0285 

00:43:34,203 INFO For client 1, upload iter 1, epoch 1, Batch 2/17, Loss:2.4725 | TotalTime 0.1003 | Comptime: 0.0265 

Terminated
