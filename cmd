screen -S jk_drop
script .log/gcn_drop8.txt
CUDA_VISIBLE_DEVICES=5 sh ./script/reddit_GCN.sh 
exit
