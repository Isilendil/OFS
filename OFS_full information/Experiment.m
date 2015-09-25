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
    %fprintf(1,'running on the %d-th trial...\n',i);
    ID = ID_list(i,:);

    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = RND(X, Y, options, ID);
    %fprintf(1,'Random selection: The number of mistakes = %d\n', err_count);
    nSV_RN(i) = length(classifier.SV);
    err_RN(i) = err_count;
    time_RN(i) = run_time;
    mistakes_list_RN(i,:) = mistakes;
    SVs_RN(i,:) = SVs;
    TMs_RN(i,:) = TMs;

    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = perceptron(X, Y,options,ID);
    %fprintf(1,'Perceptron: The number of mistakes = %d\n', err_count);
    nSV_PE(i) = length(classifier.SV);
    err_PE(i) = err_count;
    time_PE(i) = run_time;
    mistakes_list_PE(i,:) = mistakes;
    SVs_PE(i,:) = SVs;
    TMs_PE(i,:) = TMs;

    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = OFSGD(X, Y,options,ID);
    %fprintf(1,'OFSD: The number of mistakes = %d\n', err_count);
    nSV_ODGD(i) = length(classifier.SV);
    err_ODGD(i) = err_count;
    time_ODGD(i) = run_time;
    mistakes_list_ODGD(i,:) = mistakes;
    SVs_ODGD(i,:) = SVs;
    TMs_ODGD(i,:) = TMs;

end

mistakes_idx = 1:length(mistakes_idx);
%% print and plot results
figure
figure_FontSize=12;
mean_mistakes_RN = mean(mistakes_list_RN);
semilogx(2.^mistakes_idx, mean_mistakes_RN,'k.-');
hold on
mean_mistakes_PE = mean(mistakes_list_PE);
semilogx(2.^mistakes_idx, mean_mistakes_PE,'b-s');
mean_mistakes_ODGD = mean(mistakes_list_ODGD);
semilogx(2.^mistakes_idx, mean_mistakes_ODGD,'g-x');
legend('RAND_{sele}','PE_{trun}','OFS');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
grid

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'rndslc: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_RN), std(err_RN), mean(nSV_RN), std(nSV_RN), mean(time_RN), std(time_RN));
fprintf(1,'Perceptron: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PE), std(err_PE), mean(nSV_PE), std(nSV_PE), mean(time_PE), std(time_PE));
fprintf(1,'OFSGD: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(1,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_ODGD), std(err_ODGD), mean(nSV_ODGD), std(nSV_ODGD), mean(time_ODGD), std(time_ODGD));
fprintf(1,'-------------------------------------------------------------------------------\n');

