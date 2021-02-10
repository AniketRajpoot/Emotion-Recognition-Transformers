%% clear all/ close figs
close all
clear
clc

%% default paramters
Participant = 32;
Video = 40;
Channel = 32;
Fs = 128;
Time = 63;
addpath('D:\Arjun\BCI PROJECT\Emotion\deap-pre-mat')

%% set parameters
frameNum = 64;
totalScale = 48;
wname = 'coif5';
pieces = 4;

%%

for participant = 1:Participant
    %fprintf('\nworking on file number %d:\n', participant);
    if(participant<10)
        myfilename = sprintf('s0%d.mat', participant);
    else
        myfilename = sprintf('s%d.mat', participant);
    end
    load(myfilename);
    
    for video=1:Video
        
        fprintf('\ncreating file participant %d,video %d:\n',participant,video);
        op1 = 'participant';
        op2 = 'video';
        filename = [op1 int2str(participant) op2  int2str(video) '.mat'];filename;
        %fid = fopen( filename, 'wt' );
        %fprintf(filename);
        output = zeros(32, 4, 48, 48);                
        datastart=384;
        datalength=7680;
      
        
        for channel = 1:32
           for piece = 1:4
            data1=zeros(1,1920);
            dataa=1920*piece;
            iii = 1;
            for ii = dataa-1920+1:dataa
                data1(1,iii)=data(video,channel,ii+datastart);
                iii = iii + 1;
            end
   
            % decompose into wavelets
            % set scales
            f = 1:totalScale;
            f = Fs/totalScale * f;
            wcf = centfrq(wname);
            scal =  Fs * wcf./ f;           
            
            coefs = cwt(data1, scal, wname);
            coefs = imresize(coefs, [48,48]);
            %fprintf('size(A) is %s\n', mat2str(size(coefs)));
            
            output(channel, piece, :, :) = coefs;
            
           end 
        end
        save(filename,'output');
        
    end %the testcase loop
end %the file loop
