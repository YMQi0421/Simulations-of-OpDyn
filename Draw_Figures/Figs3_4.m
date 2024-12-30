close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Results I  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('SF_shannon_competition.mat')
load('SF_states_competition.mat')
% tasktype=3 2 2
Num_task=7;
Num_agent=5000;

T = 300;
t = 0;
dt = 0.5;
t_seq = t:dt:T;


% state_first_task = state_discrete_full(1:2:length(t_seq), :, 1:3);

% figure(1)
% subplot(3,1,1)
% plot(t_seq(1:2:end),state_first_task(:,:,1));
% subplot(3,1,2)
% plot(t_seq(1:2:end),state_first_task(:,:,2));
% subplot(3,1,3)
% plot(t_seq(1:2:end),state_first_task(:,:,3));
% 
% state_sec_task = state_discrete_full(1:2:length(t_seq), :, 4:5);
% 
% figure(2)
% subplot(2,1,1)
% plot(t_seq(1:2:end),state_sec_task(:,:,1));
% subplot(2,1,2)
% plot(t_seq(1:2:end),state_sec_task(:,:,2));
% 
% state_thr_task = state_discrete_full(1:2:length(t_seq), :, 6:7);
% 
% figure(3)
% subplot(2,1,1)
% plot(t_seq(1:2:end),state_thr_task(:,:,1));
% subplot(2,1,2)
% plot(t_seq(1:2:end),state_thr_task(:,:,2));

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shannon value
figure(4)
plot(t_seq,Shannon_value);hold on;
plot(t_seq,Shannon_action_value,'--');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% consensus error
state_full_sign = sign(state_discrete_full);
state_full_sign(state_full_sign<0)=0;
cluster_opinion_all=sum(state_full_sign,2);
cluster_opinion_final=cluster_opinion_all(end,:);

state_full_ave=sum(state_discrete_full,2)/Num_agent;
state_consensus_error=vecnorm(state_discrete_full-repmat(state_full_ave,1,Num_agent),2,3);
state_error = sum(abs(state_consensus_error),2)/Num_agent;

action_current_temp=action_discrete_full(end,:,:);
action_current = reshape(action_current_temp,Num_agent,Num_task);
unique_cluster=unique(action_current,'rows');
Num_clusters=size(unique_cluster,1);
for cnt_cluster=1:Num_clusters
    cluster_index=find(ismember(action_current, unique_cluster(cnt_cluster,:),'rows'));
    state_cluster_ave=sum(state_discrete_full(:,cluster_index,:),2)/length(cluster_index);
    consensus_error_cluster=vecnorm(state_discrete_full(:,cluster_index,:)-...
        repmat(state_cluster_ave,1,length(cluster_index)),2,3);
    consensus_error_cluster_full(:,cnt_cluster)=sum(consensus_error_cluster,2)/length(cluster_index);
%     [max_a,index]=max(consensus_error_cluster,[],2);
%     consensus_error_cluster_full(:,cnt_cluster)=max_a;
end



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% disagreement error
A = WeightedMatrix;
A(A>0)=1;
L=diag(sum(A,1))-A;
for cnt_time = 1:length(t_seq)
    disMatrix(cnt_time,:,:) = L * reshape(state_discrete_full(cnt_time,:,:),Num_agent,Num_task);
end

disError=vecnorm(disMatrix,2,3);
disError_full = sum(abs(disError),2)/Num_agent;

figure(9)
plot(t_seq,state_error(1:length(t_seq)),'LineWidth',2);hold on;
plot(t_seq,disError_full(1:length(t_seq)),'-.','LineWidth',2);
plot(t_seq,consensus_error_cluster_full(1:length(t_seq),:),'--');hold on;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% distribution
edges = -2:0.5:2;
Y = edges(2:length(edges));
X = 0:0.5:300.5;
% X = 1:size(state_discrete_full,1);
[XX, YY] = meshgrid(X, Y);
figure(5)
cnt_plot=1;
for cnt_distribution_task = 1:3
    subplot(3,2,cnt_plot)
    state_single_task = state_discrete_full(:,:,cnt_distribution_task);
    distribution = zeros(size(state_discrete_full,1), length(edges)-1);
    for i = 1:size(state_discrete_full,1)
        distribution(i,:) = histcounts(state_single_task(i,:,:), edges);
    end
    surf(XX,YY,distribution', 'EdgeColor','none','FaceAlpha', 0.4);
    title('3-D view');
    subplot(3,2,cnt_plot+1)
    surf(XX,YY,distribution', 'EdgeColor','none','FaceAlpha', 0.4),view(0,90)
    title('Top view')
    cnt_plot=cnt_plot+2;
end

figure(6)
cnt_plot=1;
for cnt_distribution_task = 1:2
    subplot(2,2,cnt_plot)
    state_single_task = state_discrete_full(:,:,3+cnt_distribution_task);
    distribution = zeros(size(state_discrete_full,1), length(edges)-1);
    for i = 1:size(state_discrete_full,1)
        distribution(i,:) = histcounts(state_single_task(i,:,:), edges);
    end
    surf(XX,YY,distribution', 'EdgeColor','none','FaceAlpha', 0.4);
    title('3-D view');
    subplot(2,2,cnt_plot+1)
    surf(XX,YY,distribution', 'EdgeColor','none','FaceAlpha', 0.4),view(0,90)
    title('Top view')
    cnt_plot=cnt_plot+2;
end

figure(7)
cnt_plot=1;
for cnt_distribution_task = 1:2
    subplot(2,2,cnt_plot)
    state_single_task = state_discrete_full(:,:,5+cnt_distribution_task);
    distribution = zeros(size(state_discrete_full,1), length(edges)-1);
    for i = 1:size(state_discrete_full,1)
        distribution(i,:) = histcounts(state_single_task(i,:,:), edges);
    end
    surf(XX,YY,distribution', 'EdgeColor','none','FaceAlpha', 0.4);
    title('3-D view');
    subplot(2,2,cnt_plot+1)
    surf(XX,YY,distribution', 'EdgeColor','none','FaceAlpha', 0.4),view(0,90)
    title('Top view')
    cnt_plot=cnt_plot+2;
end


