00:41:06,884 INFO tensor([0.5749], device='cuda:0')
00:41:06,889 INFO ===== Experiment start =====
00:41:17,878 INFO ====Initiating DataPartitioner takes 1.5222423076629639 s

00:43:28,757 INFO ====Initiating DataPartitioner takes 0.18977761268615723 s

00:43:29,298 INFO ========= Start of Random Partition =========

00:43:30,698 INFO Raw class per worker is : array([[56., 82., 67., ...,  0.,  0.,  0.],
       [52., 67., 57., ...,  0.,  0.,  0.],
       [46., 74., 53., ...,  0.,  0.,  0.],
       ...,
       [69., 58., 62., ...,  0.,  0.,  0.],
       [51., 59., 58., ...,  0.,  0.,  0.],
       [54., 81., 49., ...,  0.,  1.,  0.]])

00:43:30,699 INFO ========= End of Class/Worker =========

00:43:30,704 INFO ====Data length for client 3 is 1901
00:43:31,16 INFO 
Namespace(backend='nccl', batch_size=32, client_path='/tmp/client.cfg', data_dir='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg', data_mapfile='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/openImg/imageToAuthor', data_set='openImg', decay_epoch=50.0, decay_factor=0.9, depth=18, display_step=20, dump_epoch=500, duplicate_data=1, epochs=20000, eval_interval=25, eval_interval_prior=9999999, filter_class=0, filter_less=140, force_read=False, full_gradient_interval=20, gpu_device=0, hetero_allocation='1.0-1.0-1.0-1.0-1.0-1.0', heterogeneity=1.0, input_dim=0, is_even_avg=True, learners='1-2-3-4-5-6-7-8-9-10', learning_rate=0.005, load_model=False, manager_port=9005, model='squeezenet1_1', model_avg=True, num_class=596, num_loaders=2, output_dim=0, proxy_avg=False, ps_ip='10.255.11.92', ps_port='29501', resampling_interval=1, sample_mode='bandit', sample_seed=233, save_path='./', score_mode='loss', sequential='0', single_sim=0, sleep_up=0, stale_threshold=0, test_bsz=256, test_interval=999999, this_rank=4, threads='40', time_stamp='0314_004054', timeout=9999999, to_device='cuda', total_worker=100, upload_epoch=20, validate_interval=999999, zipf_alpha='5')

00:43:31,41 INFO ====Data length for client 3 is 116
00:43:32,664 INFO For client 4, upload iter 1, epoch 0, Batch 1/4, Loss:6.4487 | TotalTime 1.5967 | Comptime: 0.8583 

00:43:32,693 INFO For client 4, upload iter 1, epoch 0, Batch 2/4, Loss:6.243 | TotalTime 0.0283 | Comptime: 0.0254 

00:43:32,721 INFO For client 4, upload iter 1, epoch 0, Batch 3/4, Loss:5.4203 | TotalTime 0.0284 | Comptime: 0.0259 

00:43:32,748 INFO For client 4, upload iter 1, epoch 1, Batch 4/4, Loss:5.0808 | TotalTime 0.0265 | Comptime: 0.024 

00:43:32,750 INFO ====Data length for client 3 is 116
00:43:33,72 INFO For client 4, upload iter 1, epoch 1, Batch 0/4, Loss:5.227 | TotalTime 0.3238 | Comptime: 0.0307 

00:43:33,172 INFO For client 4, upload iter 1, epoch 1, Batch 1/4, Loss:4.884 | TotalTime 0.0989 | Comptime: 0.0285 

00:43:33,211 INFO For client 4, upload iter 1, epoch 1, Batch 2/4, Loss:5.4141 | TotalTime 0.0388 | Comptime: 0.0225 

00:43:33,234 INFO For client 4, upload iter 1, epoch 2, Batch 3/4, Loss:4.6011 | TotalTime 0.0234 | Comptime: 0.021 

00:43:33,236 INFO ====Data length for client 3 is 116
00:43:33,591 INFO For client 4, upload iter 1, epoch 2, Batch 4/4, Loss:4.9268 | TotalTime 0.3567 | Comptime: 0.0265 

00:43:33,648 INFO For client 4, upload iter 1, epoch 2, Batch 0/4, Loss:4.5475 | TotalTime 0.0558 | Comptime: 0.0233 

00:43:33,670 INFO For client 4, upload iter 1, epoch 2, Batch 1/4, Loss:4.7743 | TotalTime 0.0224 | Comptime: 0.0199 

00:43:33,735 INFO For client 4, upload iter 1, epoch 3, Batch 2/4, Loss:4.0241 | TotalTime 0.0643 | Comptime: 0.0244 

00:43:33,737 INFO ====Data length for client 3 is 116
00:43:34,95 INFO For client 4, upload iter 1, epoch 3, Batch 3/4, Loss:4.4984 | TotalTime 0.3594 | Comptime: 0.029 

00:43:34,127 INFO For client 4, upload iter 1, epoch 3, Batch 4/4, Loss:4.4434 | TotalTime 0.032 | Comptime: 0.0248 

00:43:34,150 INFO For client 4, upload iter 1, epoch 3, Batch 0/4, Loss:4.5661 | TotalTime 0.0227 | Comptime: 0.02 

00:43:34,262 INFO For client 4, upload iter 1, epoch 4, Batch 1/4, Loss:4.3757 | TotalTime 0.1117 | Comptime: 0.0243 

00:43:34,267 INFO ====Data length for client 3 is 116
00:43:34,595 INFO For client 4, upload iter 1, epoch 4, Batch 2/4, Loss:4.8314 | TotalTime 0.3288 | Comptime: 0.0288 

00:43:34,652 INFO For client 4, upload iter 1, epoch 4, Batch 3/4, Loss:4.3463 | TotalTime 0.0566 | Comptime: 0.03 

00:43:34,680 INFO For client 4, upload iter 1, epoch 4, Batch 4/4, Loss:4.0646 | TotalTime 0.0283 | Comptime: 0.0259 

00:43:34,821 INFO For client 4, upload iter 1, epoch 5, Batch 0/4, Loss:4.4179 | TotalTime 0.1411 | Comptime: 0.0294 

Terminated
