for i in {1..5}
do 
    CUDA_VISIBLE_DEVICES=0,1 bash run.sh configs/GlaS/GlaS_R_18_FPN_1x_moco.yaml \
        ./work_dirs/converted_weights/CA2CL.pkl  \
        2  \
        work_dirs/wsi_submission/GlaS/GlaS_R_18_FPN_0.5x_moco/$i
done

for i in {1..5}
do 
    CUDA_VISIBLE_DEVICES=0,1 bash run.sh configs/CRAG/CRAG_R_18_FPN_1x_moco.yaml \
        ./work_dirs/converted_weights/CA2CL.pkl  \
        2  \
        work_dirs/wsi_submission/CRAG/CRAG_R_18_FPN_0.5x_moco/$i
done