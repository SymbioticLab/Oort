00:41:06,859 INFO tensor([0.9126], device='cuda:0')
00:41:06,864 INFO ===== Experiment start =====
00:41:16,939 INFO ====Initiating DataPartitioner takes 1.533193588256836 s

00:43:28,755 INFO ====Initiating DataPartitioner takes 0.18782877922058105 s

00:43:29,298 INFO ========= Start of Random Partition =========

00:43:30,716 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,716 INFO ========= End of Class/Worker =========

00:43:30,722 INFO ====Data length for client 5 is 1901
00:43:30,963 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=6, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:30,987 INFO ====Data length for client 5 is 10
00:43:32,22 INFO For client 6, upload iter 1, epoch 1, Batch 1/1, Loss:6.3775 | TotalTime 1.0093 | Comptime: 0.6867 

00:43:32,23 INFO ====Data length for client 5 is 10
00:43:32,258 INFO For client 6, upload iter 1, epoch 2, Batch 0/1, Loss:5.9718 | TotalTime 0.2353 | Comptime: 0.0275 

00:43:32,259 INFO ====Data length for client 5 is 10
00:43:32,551 INFO For client 6, upload iter 1, epoch 3, Batch 1/1, Loss:4.4624 | TotalTime 0.2922 | Comptime: 0.0306 

00:43:32,552 INFO ====Data length for client 5 is 10
00:43:32,831 INFO For client 6, upload iter 1, epoch 4, Batch 0/1, Loss:6.755 | TotalTime 0.2794 | Comptime: 0.0306 

00:43:32,832 INFO ====Data length for client 5 is 10
00:43:33,109 INFO For client 6, upload iter 1, epoch 5, Batch 1/1, Loss:4.7889 | TotalTime 0.2775 | Comptime: 0.0289 

00:43:33,110 INFO ====Data length for client 5 is 10
00:43:33,381 INFO For client 6, upload iter 1, epoch 6, Batch 0/1, Loss:5.7162 | TotalTime 0.2714 | Comptime: 0.031 

00:43:33,383 INFO ====Data length for client 5 is 10
00:43:33,641 INFO For client 6, upload iter 1, epoch 7, Batch 1/1, Loss:6.003 | TotalTime 0.2593 | Comptime: 0.0315 

00:43:33,643 INFO ====Data length for client 5 is 10
00:43:33,909 INFO For client 6, upload iter 1, epoch 8, Batch 0/1, Loss:5.8061 | TotalTime 0.2674 | Comptime: 0.0293 

00:43:33,911 INFO ====Data length for client 5 is 10
00:43:34,171 INFO For client 6, upload iter 1, epoch 9, Batch 1/1, Loss:4.9768 | TotalTime 0.2615 | Comptime: 0.0314 

00:43:34,173 INFO ====Data length for client 5 is 10
00:43:34,463 INFO For client 6, upload iter 1, epoch 10, Batch 0/1, Loss:5.9967 | TotalTime 0.2905 | Comptime: 0.0326 

00:43:34,464 INFO ====Data length for client 5 is 10
00:43:34,741 INFO For client 6, upload iter 1, epoch 11, Batch 1/1, Loss:4.3242 | TotalTime 0.2774 | Comptime: 0.0308 

00:43:34,742 INFO ====Data length for client 5 is 10
00:43:35,19 INFO For client 6, upload iter 1, epoch 12, Batch 0/1, Loss:4.4372 | TotalTime 0.2776 | Comptime: 0.0292 

00:43:35,24 INFO ====Data length for client 5 is 10
00:43:35,311 INFO For client 6, upload iter 1, epoch 13, Batch 1/1, Loss:3.7597 | TotalTime 0.2884 | Comptime: 0.0312 

00:43:35,313 INFO ====Data length for client 5 is 10
00:43:35,578 INFO For client 6, upload iter 1, epoch 14, Batch 0/1, Loss:4.1536 | TotalTime 0.2662 | Comptime: 0.0281 

00:43:35,580 INFO ====Data length for client 5 is 10
00:43:35,838 INFO For client 6, upload iter 1, epoch 15, Batch 1/1, Loss:3.6525 | TotalTime 0.2594 | Comptime: 0.0282 

00:43:35,840 INFO ====Data length for client 5 is 10
00:43:36,143 INFO For client 6, upload iter 1, epoch 16, Batch 0/1, Loss:3.6783 | TotalTime 0.3038 | Comptime: 0.0327 

00:43:36,144 INFO ====Data length for client 5 is 10
00:43:36,418 INFO For client 6, upload iter 1, epoch 17, Batch 1/1, Loss:3.5595 | TotalTime 0.275 | Comptime: 0.0289 

00:43:36,420 INFO ====Data length for client 5 is 10
00:43:36,682 INFO For client 6, upload iter 1, epoch 18, Batch 0/1, Loss:3.726 | TotalTime 0.263 | Comptime: 0.0322 

00:43:36,684 INFO ====Data length for client 5 is 10
00:43:36,961 INFO For client 6, upload iter 1, epoch 19, Batch 1/1, Loss:3.6331 | TotalTime 0.2785 | Comptime: 0.0318 

00:43:36,963 INFO ====Data length for client 5 is 10
00:43:37,229 INFO For client 6, upload iter 1, epoch 20, Batch 0/1, Loss:3.7093 | TotalTime 0.2668 | Comptime: 0.0295 

Terminated
