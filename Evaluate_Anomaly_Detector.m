clc
clear all
close all
 

C3D_CNN_Path='/home/nit/Documents/BeforeSave32'; % C3D features for videos
Testing_VideoPath='/home/nit/Documents/Testing_Videos'; % Path of mp4 videos
AllAnn_Path='/home/nit/Downloads/dataset/Temporal_Anomaly_Annotation_For_Testing_Videos/Matlab_formate'; % Path of Temporal Annotations
Model_Score_Folder='/home/nit/AnomalyDetectionCVPR2018/output2';  % Path of Pretrained Model score on Testing videos (32 numbers for 32 temporal segments)
Paper_Results='/home/nit/AnomalyDetectionCVPR2018/output4';   % Path to save results.


All_Videos_scores=dir(Model_Score_Folder);
Testing_Videos1=dir(AllAnn_Path);
Testing_Videos1=Testing_Videos1(3:end);
All_Videos_scores=All_Videos_scores(3:end);
nVideos=length(All_Videos_scores);
frm_counter=1;
All_Detect=zeros(1,1000000);
All_GT=zeros(1,1000000);

for ivideo=1:nVideos
    ivideo
    All_Videos_scores(ivideo).name(1:end-9)        
%   Testing_Videos1(ivideo).name(1:end-3)
    Ann_Path=[AllAnn_Path,'/',All_Videos_scores(ivideo).name(1:end-10),'.mat'];
    load(Ann_Path)
%    check=strmatch(All_Videos_scores(ivideo).name(1:end-9),Testing_Videos1(ivideo).name(1:end-3));
%    if isempty(check)
%         error('????') 
%    end
    %All_Videos_scores(ivideo).name(1:end-10)
    VideoPath=[Testing_VideoPath,'/', All_Videos_scores(ivideo).name(1:end-10),'.mp4'];
    %VideoPath=VideoPath.name(1:end-6)
    VideoPath
    ScorePath=[Model_Score_Folder,'/', All_Videos_scores(ivideo).name(1:end-10),'.mp4_C.mat'];
    ScorePath

  %% Load Video
    try
        xyloObj = VideoReader(VideoPath);
        xyloObj
    catch
       error('???')
    end

    Predic_scores=load(ScorePath);
    fps=30;
    Actual_frames=round(xyloObj.Duration*fps);
    Actual_frames
    Folder_Path=[C3D_CNN_Path,'/',All_Videos_scores(ivideo).name(1:end-6)];
    Folder_Path
    AllFiles=dir([Folder_Path,'/*.fc6-1']);
    nFileNumbers=length(AllFiles);
    nFileNumbers
    nFrames_C3D=nFileNumbers*16;  % As the features were computed for every 16 frames
    nFrames_C3D
    


%% 32 Shots
    Detection_score_32shots=zeros(1,nFrames_C3D);
    Thirty2_shots= round(linspace(1,length(AllFiles),33));
    Thirty2_shots
    Shots_Features=[];
    p_c=0;

    for ishots=1:length(Thirty2_shots)-1

        p_c=p_c+1;
        ss=Thirty2_shots(ishots);
        ee=Thirty2_shots(ishots+1)-1;

        if ishots==length(Thirty2_shots)
            ee=Thirty2_shots(ishots+1);
        end
        %p_c
        if ee<ss
            Detection_score_32shots((ss-1)*16+1:(ss-1)*16+1+15)=Predic_scores.x(p_c);   
        else
            Detection_score_32shots((ss-1)*16+1:(ee-1)*16+16)=Predic_scores.x(p_c);
        end

    end


    Final_score=  [Detection_score_32shots,repmat(Detection_score_32shots(end),[1,Actual_frames-length(Detection_score_32shots)])];
    GT=zeros(1,Actual_frames);
        
    Testing_Videos1=[AllAnn_Path,'/',All_Videos_scores(ivideo).name(1:end-9),'mat'];
    Testing_Videos1
    Testing_Videos1=load(Testing_Videos1)
    Testing_Videos1.Annotation_file.Anno(1,1)
    for ik=1:size(Testing_Videos1.Annotation_file.Anno,1)
            st_fr=max(Testing_Videos1.Annotation_file.Anno(ik,1),1); 
            end_fr=min(Testing_Videos1.Annotation_file.Anno(ik,2),Actual_frames);
            GT(st_fr:end_fr)=1;
    end


    if Testing_Videos1.Annotation_file.Anno(1,1)==0.05   % For Normal Videos
        GT=zeros(1,Actual_frames);
    end


    %Final_score= ones(1,length(Final_score));
    %subplot(2,1,1); bar(Final_score)
    %subplot(2,1,2); bar(GT)

    All_Detect(frm_counter:frm_counter+length(Final_score)-1)=Final_score;
    All_GT(frm_counter:frm_counter+length(Final_score)-1)=GT;
    frm_counter=frm_counter+length(Final_score);


end


All_Detect=(All_Detect(1:frm_counter-1));
All_GT=All_GT(1:frm_counter-1);
scores=All_Detect;
[so,si] = sort(scores,'descend');
tp=All_GT(si)>0;
fp=All_GT(si)==0;
tp=cumsum(tp);
fp=cumsum(fp);
nrpos=sum(All_GT);
rec=tp/nrpos;
fpr=fp/sum(All_GT==0);
prec=tp./(fp+tp);
AUC1 = trapz(fpr ,rec );
%You can also use the following codes
[X,Y,T,AUC] = perfcurve(All_GT,All_Detect,1);

 
