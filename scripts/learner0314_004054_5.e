00:41:06,832 INFO tensor([0.4690], device='cuda:0')
00:41:06,837 INFO ===== Experiment start =====
00:41:16,874 INFO ====Initiating DataPartitioner takes 1.5367178916931152 s

00:43:28,757 INFO ====Initiating DataPartitioner takes 0.18865537643432617 s

00:43:29,302 INFO ========= Start of Random Partition =========

00:43:30,699 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,699 INFO ========= End of Class/Worker =========

00:43:30,705 INFO ====Data length for client 4 is 1901
00:43:30,986 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=5, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:31,12 INFO ====Data length for client 4 is 454
00:43:32,513 INFO For client 5, upload iter 1, epoch 0, Batch 1/15, Loss:6.4272 | TotalTime 1.4744 | Comptime: 1.1388 

00:43:32,541 INFO For client 5, upload iter 1, epoch 0, Batch 2/15, Loss:6.3263 | TotalTime 0.028 | Comptime: 0.0252 

00:43:32,568 INFO For client 5, upload iter 1, epoch 0, Batch 3/15, Loss:5.4547 | TotalTime 0.0268 | Comptime: 0.0242 

00:43:32,592 INFO For client 5, upload iter 1, epoch 0, Batch 4/15, Loss:5.1918 | TotalTime 0.0229 | Comptime: 0.0201 

00:43:32,617 INFO For client 5, upload iter 1, epoch 0, Batch 5/15, Loss:4.6517 | TotalTime 0.0249 | Comptime: 0.022 

00:43:32,641 INFO For client 5, upload iter 1, epoch 0, Batch 6/15, Loss:3.6182 | TotalTime 0.0235 | Comptime: 0.0206 

00:43:32,668 INFO For client 5, upload iter 1, epoch 0, Batch 7/15, Loss:5.5321 | TotalTime 0.0271 | Comptime: 0.0245 

00:43:32,693 INFO For client 5, upload iter 1, epoch 0, Batch 8/15, Loss:4.2297 | TotalTime 0.0251 | Comptime: 0.0222 

00:43:32,750 INFO For client 5, upload iter 1, epoch 0, Batch 9/15, Loss:4.2591 | TotalTime 0.0565 | Comptime: 0.0291 

00:43:32,822 INFO For client 5, upload iter 1, epoch 0, Batch 10/15, Loss:4.2239 | TotalTime 0.0711 | Comptime: 0.0254 

00:43:32,881 INFO For client 5, upload iter 1, epoch 0, Batch 11/15, Loss:4.1288 | TotalTime 0.0592 | Comptime: 0.0242 

00:43:32,939 INFO For client 5, upload iter 1, epoch 0, Batch 12/15, Loss:4.3251 | TotalTime 0.0583 | Comptime: 0.0219 

00:43:32,999 INFO For client 5, upload iter 1, epoch 0, Batch 13/15, Loss:4.1662 | TotalTime 0.0597 | Comptime: 0.0284 

00:43:33,25 INFO For client 5, upload iter 1, epoch 0, Batch 14/15, Loss:2.9185 | TotalTime 0.0251 | Comptime: 0.0231 

00:43:33,100 INFO For client 5, upload iter 1, epoch 1, Batch 15/15, Loss:6.8026 | TotalTime 0.0753 | Comptime: 0.0272 

00:43:33,105 INFO ====Data length for client 4 is 454
00:43:33,421 INFO For client 5, upload iter 1, epoch 1, Batch 0/15, Loss:5.6447 | TotalTime 0.3208 | Comptime: 0.0333 

00:43:33,448 INFO For client 5, upload iter 1, epoch 1, Batch 1/15, Loss:6.0485 | TotalTime 0.0226 | Comptime: 0.0201 

00:43:33,523 INFO For client 5, upload iter 1, epoch 1, Batch 2/15, Loss:6.0869 | TotalTime 0.0752 | Comptime: 0.0237 

00:43:33,552 INFO For client 5, upload iter 1, epoch 1, Batch 3/15, Loss:6.1227 | TotalTime 0.0278 | Comptime: 0.025 

00:43:33,598 INFO For client 5, upload iter 1, epoch 1, Batch 4/15, Loss:6.0502 | TotalTime 0.0463 | Comptime: 0.0277 

Terminated
