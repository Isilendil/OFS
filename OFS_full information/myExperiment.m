function Experiment(dataset_name,data)

%% load dataset
load(sprintf('data/%s',dataset_name));
[n,d]       = size(data);
ID_list = ID_ALL;
Y = data(1:n,1);
X = data(1:n,2:d);


stdX=std(X);
idx1=stdX~=0;
centrX=X-repmat(mean(X),size(X,1),1);
X(:,idx1)=centrX(:,idx1)./repmat(stdX(:,idx1),size(X,1),1);

X=(X-repmat(mean(X),size(X,1),1))./repmat(std(X),size(X,1),1);
X=X./repmat(sqrt(sum(X.*X,2)),1, size(X,2));

data=[Y,X];
options.t_tick=round(n/15);
options.NumFeature=max(1,round(0.1*(d-1)));


%% run experiments:
for i=1:20,
    fprintf(1,'running on the %d-th trial...\n',i);
    ID = ID_list(i,:);
    
    %OFS
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = OFSGD(X, Y,options,ID);
    mistakes_list_ODGD(i,:) = mistakes;
    
    %PA
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA(X, Y,options,ID);
    mistakes_list_PA(i,:) = mistakes;

    
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA_truncate(X, Y,options,ID);
    mistakes_list_PA_truncate(i,:) = mistakes;

    
    %PA1
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA1(X, Y,options,ID);
    mistakes_list_PA1(i,:) = mistakes;

    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA1_truncate(X, Y,options,ID);
    mistakes_list_PA1_truncate(i,:) = mistakes;

    
     %PA2
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA2(X, Y,options,ID);
    mistakes_list_PA2(i,:) = mistakes;

    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA2_truncate(X, Y,options,ID);
    mistakes_list_PA2_truncate(i,:) = mistakes;

end

mistakes_idx = 1:length(mistakes_idx);
%% print and plot results
figure
figure_FontSize=12;

mean_mistakes_ODGD = mean(mistakes_list_ODGD);
semilogx(2.^mistakes_idx, mean_mistakes_ODGD,'b-x');
hold on

mean_mistakes_PA = mean(mistakes_list_PA);
semilogx(2.^mistakes_idx, mean_mistakes_PA,'r-x');

mean_mistakes_PA_truncate = mean(mistakes_list_PA_truncate);
semilogx(2.^mistakes_idx, mean_mistakes_PA_truncate,'m-x');

mean_mistakes_PA1 = mean(mistakes_list_PA1);
semilogx(2.^mistakes_idx, mean_mistakes_PA1,'y-x');

mean_mistakes_PA1_truncate = mean(mistakes_list_PA1_truncate);
semilogx(2.^mistakes_idx, mean_mistakes_PA1_truncate,'c-x');

mean_mistakes_PA2 = mean(mistakes_list_PA2);
semilogx(2.^mistakes_idx, mean_mistakes_PA2,'g-x');

mean_mistakes_PA2_truncate = mean(mistakes_list_PA2_truncate);
semilogx(2.^mistakes_idx, mean_mistakes_PA2_truncate, 'k-x');

title(dataset_name)
legend('OFS', 'PA', 'PA_np', 'PA1', 'PA1_np', 'PA2', 'PA2_np');

xlabel('Number of samples');
ylabel('Online average rate of mistakes')
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
grid
%print(1,'-jpg',['./ExperimentResult/' dataset_name '.jpg']);

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'OFSGD: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_ODGD), std(err_ODGD), mean(nSV_ODGD), std(nSV_ODGD), mean(time_ODGD), std(time_ODGD));
fprintf(1,'PA: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA), std(err_PA), mean(nSV_PA), std(nSV_PA), mean(time_PA), std(time_PA));
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA_truncate), std(err_PA_truncate), mean(nSV_PA_truncate), std(nSV_PA_truncate), mean(time_PA_truncate), std(time_PA_truncate));
fprintf(1,'PA1: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA1), std(err_PA1), mean(nSV_PA1), std(nSV_PA1), mean(time_PA1), std(time_PA1));
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA1_truncate), std(err_PA1_truncate), mean(nSV_PA1_truncate), std(nSV_PA1_truncate), mean(time_PA1_truncate), std(time_PA1_truncate));
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA1), std(err_PA2), mean(nSV_PA2), std(nSV_PA2), mean(time_PA2), std(time_PA2));
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA1_truncate), std(err_PA2_truncate), mean(nSV_PA2_truncate), std(nSV_PA2_truncate), mean(time_PA2_truncate), std(time_PA2_truncate));
fprintf(1,'-------------------------------------------------------------------------------\n');


save(['./ExperimentResult/' dataset_name], 'mistakes_list_ODGD', 'mistakes_list_PA', 'mistakes_list_PA1', 'mistakes_list_PA2', 'mistakes_list_PA_truncate', 'mistakes_list_PA1_truncate', 'mistakes_list_PA2_truncate')
