function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = RND(X, Y, options, id_list)
NumFeature=options.NumFeature;

ID = id_list;
err_count = 0;
t_tick = options.t_tick;
mistakes = [];
mistakes_idx = [];
SV = [];
SVs = [];
TMs=[];
k = 2;

w=zeros(size(X,2),1);     % initialize the weight vector
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    %% prediction
    x_t=X(id,:)';
    f_t=w'*x_t;

    %% feedback
    y_t=Y(id);

    if y_t*f_t<=0,
        err_count = err_count + 1;

        w=w+y_t*x_t;
        %% random selection of features
        v_idx=zeros(size(w,1),1);
        perm_t=randperm(size(w,1));
        c_t=perm_t(1:NumFeature);
        v_idx(c_t)=1;

        w=w.*v_idx;
        SV = [SV id];
    end

    run_time = toc;
    if (t==k)
        k = 2*k;
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end

classifier.SV = SV;
classifier.w = w;
run_time = toc;

