00:41:08,670 INFO tensor([0.1197], device='cuda:0')
00:41:08,682 INFO ===== Experiment start =====
00:41:19,313 INFO ====Initiating DataPartitioner takes 1.5559625625610352 s

00:43:28,768 INFO ====Initiating DataPartitioner takes 0.19909429550170898 s

00:43:29,312 INFO ========= Start of Random Partition =========

00:43:30,725 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,725 INFO ========= End of Class/Worker =========

00:43:30,731 INFO ====Data length for client 9 is 1901
00:43:31,12 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=10, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:31,36 INFO ====Data length for client 9 is 30
00:43:32,622 INFO For client 10, upload iter 1, epoch 1, Batch 1/1, Loss:6.3829 | TotalTime 1.5619 | Comptime: 0.8339 

00:43:32,624 INFO ====Data length for client 9 is 30
00:43:32,964 INFO For client 10, upload iter 1, epoch 2, Batch 0/1, Loss:6.2737 | TotalTime 0.3411 | Comptime: 0.0325 

00:43:32,965 INFO ====Data length for client 9 is 30
00:43:33,339 INFO For client 10, upload iter 1, epoch 3, Batch 1/1, Loss:5.8679 | TotalTime 0.374 | Comptime: 0.0324 

00:43:33,340 INFO ====Data length for client 9 is 30
00:43:33,709 INFO For client 10, upload iter 1, epoch 4, Batch 0/1, Loss:4.8906 | TotalTime 0.3695 | Comptime: 0.0328 

00:43:33,711 INFO ====Data length for client 9 is 30
00:43:34,65 INFO For client 10, upload iter 1, epoch 5, Batch 1/1, Loss:5.8457 | TotalTime 0.3553 | Comptime: 0.0295 

00:43:34,67 INFO ====Data length for client 9 is 30
00:43:34,437 INFO For client 10, upload iter 1, epoch 6, Batch 0/1, Loss:4.9123 | TotalTime 0.3713 | Comptime: 0.0308 

00:43:34,439 INFO ====Data length for client 9 is 30
00:43:34,814 INFO For client 10, upload iter 1, epoch 7, Batch 1/1, Loss:5.1727 | TotalTime 0.3758 | Comptime: 0.028 

00:43:34,815 INFO ====Data length for client 9 is 30
00:43:35,175 INFO For client 10, upload iter 1, epoch 8, Batch 0/1, Loss:4.7695 | TotalTime 0.3609 | Comptime: 0.0292 

00:43:35,177 INFO ====Data length for client 9 is 30
00:43:35,568 INFO For client 10, upload iter 1, epoch 9, Batch 1/1, Loss:5.3545 | TotalTime 0.3926 | Comptime: 0.0322 

00:43:35,570 INFO ====Data length for client 9 is 30
00:43:35,929 INFO For client 10, upload iter 1, epoch 10, Batch 0/1, Loss:4.684 | TotalTime 0.3598 | Comptime: 0.0329 

00:43:35,931 INFO ====Data length for client 9 is 30
00:43:36,286 INFO For client 10, upload iter 1, epoch 11, Batch 1/1, Loss:4.7929 | TotalTime 0.3562 | Comptime: 0.0299 

00:43:36,288 INFO ====Data length for client 9 is 30
00:43:36,643 INFO For client 10, upload iter 1, epoch 12, Batch 0/1, Loss:4.4687 | TotalTime 0.3561 | Comptime: 0.0271 

00:43:36,648 INFO ====Data length for client 9 is 30
00:43:37,29 INFO For client 10, upload iter 1, epoch 13, Batch 1/1, Loss:4.7048 | TotalTime 0.3826 | Comptime: 0.033 

00:43:37,112 INFO ====Data length for client 9 is 30
00:43:37,498 INFO For client 10, upload iter 1, epoch 14, Batch 0/1, Loss:4.3815 | TotalTime 0.3878 | Comptime: 0.0325 

00:43:37,500 INFO ====Data length for client 9 is 30
00:43:37,865 INFO For client 10, upload iter 1, epoch 15, Batch 1/1, Loss:4.4902 | TotalTime 0.3661 | Comptime: 0.0292 

00:43:37,867 INFO ====Data length for client 9 is 30
00:43:38,235 INFO For client 10, upload iter 1, epoch 16, Batch 0/1, Loss:4.4474 | TotalTime 0.3695 | Comptime: 0.0292 

00:43:38,237 INFO ====Data length for client 9 is 30
00:43:38,605 INFO For client 10, upload iter 1, epoch 17, Batch 1/1, Loss:4.3222 | TotalTime 0.3693 | Comptime: 0.0292 

00:43:38,607 INFO ====Data length for client 9 is 30
00:43:38,965 INFO For client 10, upload iter 1, epoch 18, Batch 0/1, Loss:4.3574 | TotalTime 0.3593 | Comptime: 0.0292 

00:43:38,967 INFO ====Data length for client 9 is 30
00:43:39,349 INFO For client 10, upload iter 1, epoch 19, Batch 1/1, Loss:4.262 | TotalTime 0.3826 | Comptime: 0.0324 

00:43:39,350 INFO ====Data length for client 9 is 30
00:43:39,717 INFO For client 10, upload iter 1, epoch 20, Batch 0/1, Loss:4.2214 | TotalTime 0.3676 | Comptime: 0.0313 

Terminated
