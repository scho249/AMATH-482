clear; close all; clc;
%% unzip gz files
gunzip('train-images-idx3-ubyte.gz');
gunzip('train-labels-idx1-ubyte.gz');
gunzip('t10k-images-idx3-ubyte.gz');
gunzip('t10k-labels-idx1-ubyte.gz');
%% parse
[trnimgs, trnlbls] = mnist_parse("train-images-idx3-ubyte", "train-labels-idx1-ubyte");

[testimgs, testlbls] = mnist_parse("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

%% reshape
testdat=[]; trndat=[];
for k = 1:size(testimgs,3)
    img = reshape(testimgs(:,:,k),1,[]);
    img = img';
    testdat(:,k) = img;
end
for k = 1:size(trnimgs,3)
    img = reshape(trnimgs(:,:,k),1,[]);
    img = img';
    trndat(:,k) = img;
end

%% training
mtrn = mean(trndat,2);
trndat = trndat - repmat(mtrn,1,size(trndat,2));
[Utrn, Strn, Vtrn] = svd(trndat, 'econ');
% projection onto principal components
Ytrn = Strn*Vtrn';

%% test
testdat = testdat - repmat(mtrn,1,size(testdat,2));
[Utst, Stst, Vtst] = svd(testdat, 'econ');
% projection onto principal components
Ytst = Stst*Vtst';

%% Plotting singular values + cumulative energy
figure()
subplot(2,1,1)
plot(diag(Strn), 'ko', 'Linewidth', 1)
ylabel('\sigma')
subplot(2,1,2)
plot(cumsum(diag(Strn).^2)/sum(diag(Strn).^2),'ko','Linewidth',1)
ylabel('Cumulative Energy')

figure()
subplot(2,1,1)
plot(diag(Stst), 'ko', 'Linewidth', 1)
ylabel('\sigma')
subplot(2,1,2)
plot(cumsum(diag(Stst).^2)/sum(diag(Stst).^2),'ko','Linewidth',1)
ylabel('Cumulative Energy')

%% Plotting 
figure()
plot3(Vtrn(find(trnlbls==0),1), Vtrn(find(trnlbls==0),2), Vtrn(find(trnlbls==0),3), '.', 'color', [0 0.5 0])
hold on
plot3(Vtrn(find(trnlbls==1),1), Vtrn(find(trnlbls==1),2), Vtrn(find(trnlbls==1),3), '.', 'color', 'b')
hold on
plot3(Vtrn(find(trnlbls==2),1), Vtrn(find(trnlbls==2),2), Vtrn(find(trnlbls==2),3), '.', 'color', 'g')
hold on
plot3(Vtrn(find(trnlbls==3),1), Vtrn(find(trnlbls==3),2), Vtrn(find(trnlbls==3),3), '.', 'color', 'r')
hold on
plot3(Vtrn(find(trnlbls==4),1), Vtrn(find(trnlbls==4),2), Vtrn(find(trnlbls==4),3), '.', 'color', 'k')
hold on
plot3(Vtrn(find(trnlbls==5),1), Vtrn(find(trnlbls==5),2), Vtrn(find(trnlbls==5),3), '.', 'color', 'm')
hold on
plot3(Vtrn(find(trnlbls==6),1), Vtrn(find(trnlbls==6),2), Vtrn(find(trnlbls==6),3), '.', 'color', 'y')
hold on
plot3(Vtrn(find(trnlbls==7),1), Vtrn(find(trnlbls==7),2), Vtrn(find(trnlbls==7),3), '.', 'color', [0.2 0 0])
hold on
plot3(Vtrn(find(trnlbls==8),1), Vtrn(find(trnlbls==8),2), Vtrn(find(trnlbls==8),3), '.', 'color', [0.5 0 0.5])
hold on
plot3(Vtrn(find(trnlbls==9),1), Vtrn(find(trnlbls==9),2), Vtrn(find(trnlbls==9),3), '.', 'color', 'c')
grid on
legend(string([0:1:9]));
xlabel('mode 1'); ylabel('mode 2'); zlabel('mode 3');

%% LDA on two digits
feature = 80;
I1 = find(trnlbls == 1); I0 = find(trnlbls == 0); 
D1 = Ytrn(1:feature,I1); D0 = Ytrn(1:feature,I0); 
n1 = size(D1,2);
n0 = size(D0,2);

m1 = mean(D1,2);
m0 = mean(D0,2);

Sw = 0; % within class variances
for j = 1:n1
    Sw = Sw + (D1(:,j) - m1)*(D1(:,j) - m1)';
end
for j = 1:n0
    Sw = Sw + (D0(:,j) - m0)*(D0(:,j) - m0)';
end
Sb = (m1-m0)*(m1-m0)'; % between class

% find w
[V2, L] = eig(Sb,Sw);
[lambda, ind] = max(abs(diag(L)));
w = V2(:,ind);
w = w/norm(w,2);


% project onto w
vd1 = w'*D1;
vd0 = w'*D0;

if mean(vd0) > mean(vd1)
    w = -w;
    vd1 = -vd1;
    vd0 = -vd0;
end

sort1 = sort(vd1); 
sort0 = sort(vd0); t1 = 1; t0 = length(sort0);

while sort0(t0) > sort1(t1)
    t0 = t0 - 1;
    t1 = t1 + 1;
end
threshold = (sort0(t0) + sort1(t1))/2;

%%
figure()
plot(vd0,zeros(n0),'ob')
hold on
plot(vd1,ones(n1),'or')
yticks([0 1])
yticklabels({'digit 0', 'digit 1'})

figure()
subplot(1,2,1)
histogram(sort0,30); hold on, plot([threshold threshold], [0 1000], 'r')
title('digit 0')
subplot(1,2,2)
histogram(sort1,30); hold on, plot([threshold threshold], [0 1200], 'r')
title('digit 1')

%% 2 digits accuracy 
feature = 30;
count = 1;
success_lda = [];
for i=1:10
    for j=i+1:10
        d1 = i-1; d2 = j-1;
        T1 = find(testlbls==d1); T2 = find(testlbls==d2);
        sample = [testdat(:,T1) testdat(:,T2)];
        I1 = find(trnlbls==d1); I2 = find(trnlbls==d2);
        [U,S,V,threshold,w,sortd1,sortd2] = digit_trainer(trndat(:,I1), trndat(:,I2),feature);
        testmat = U'*sample;
        pval = w'*testmat;
        ResVec = (pval > threshold);
        err = abs(ResVec - [zeros(1,length(T1)) ones(1,length(T2))]);
        success_lda(count) = 1-sum(err)/length(ResVec);
        count = count+1;
    end
end
%% plotting success rate
[m1 m1i] = max(success_lda); [m2 m2i] = min(success_lda);
figure()
plot(success_lda,'-o'); ylim([0.9, 1]);
hold on
plot(m1i, m1,'ro')
hold on 
plot(m2i, m2,'go')
xticks(1:45)
xtickangle(90)
xticklabels(pairs)
xlabel('pairs'); ylabel('accuracy %')
title('LDA success rate for all digit pairs')

%% LDA on three digits; classification using classify function
feature = 80;
I0 = find(trnlbls == 0); I1 = find(trnlbls == 1); I2 = find(trnlbls == 4); 
Utrn_r = Utrn(:,1:feature);
D0 = Utrn_r'*trndat(:,I0); D1 = Utrn_r'*trndat(:,I1); D2 = Utrn_r'*trndat(:,I2); 
train = [D0 D1 D2];
group = [trnlbls(I0); trnlbls(I1); trnlbls(I2);];

It0 = find(testlbls == 0); It1 = find(testlbls == 1); It2 = find(testlbls == 4); 
sampled0 = Utrn_r'*testdat(:,It0);
sampled1 = Utrn_r'*testdat(:,It1);
sampled2 = Utrn_r'*testdat(:,It2);
sample = [sampled0 sampled1 sampled2];
class = classify(sample', train', group);
err = class - [testlbls(It0);testlbls(It1);testlbls(It2)];
success_3d = length(find(err==0))/length(err); %0.9709

%% decision tree multiclass
feature = 30;
count = 1;
success_tree = [];
Utrn_r = Utrn(:,1:feature);
tree=fitctree((Utrn_r'*trndat)',trnlbls);
tree_labels1 = predict(tree, (Utrn_r'*testdat)');

err_tree = abs(tree_labels1-testlbls);
length(find(err_tree==0))/length(err_tree) % 0.8522

%% decision tree pair accuracy
feature = 30;
count = 1;
success_tree = [];
Utrn_r = Utrn(:,1:feature);
for i=1:10
    for j=i+1:10
        d1 = i-1; d2 = j-1;
        I1 = find(trnlbls == d1); I2 = find(trnlbls == d2);
        D1 = Utrn_r'*trndat(:,I1); D2 = Utrn_r'*trndat(:,I2);
        train = [D1 D2];
        group = [trnlbls(I1); trnlbls(I2)];
        tree=fitctree(train',group);
        T1 = find(testlbls==d1); T2 = find(testlbls==d2);
        test = [Utrn_r*testdat(:,T1) Utrn_r*testdat(:,T2)];
        label = predict(tree, test');
        err_tree = abs(label-[testlbls(T1); testlbls(T2)]);
        success_tree(count) = length(find(err_tree==0))/length(err_tree);
        count = count+1;
    end
end
%% decision tree accuracy plot
[m1 m1i] = max(success_tree); [m2 m2i] = min(success_tree);
figure()
plot(success_tree,'-o'); ylim([0.95, 1]);
hold on
plot(m1i, m1,'ro')
hold on 
plot(m2i, m2,'go')
xticks(1:45)
xtickangle(90)
xticklabels(pairs)
xlabel('pairs'); ylabel('accuracy %')
title('Decision Tree success rate for all digit pairs')

%% multiclass svm
feature = 30;
Utrn_r = Utrn(:,1:feature);
Md1 = fitcecoc((Utrn_r'*trndat)',trnlbls);
svm_lables = predict(Md1, (Utrn_r'*testdat)');

%%
err_ecoc = abs(svm_lables-testlbls);
success_ecoc = length(find(err_ecoc==0))/length(err_ecoc) %0.4465
%% svm
feature = 30;
count = 1;
success_svm = [];
Utrn_r = Utrn(:,1:feature);
for i=1:10
    for j=i+1:10
        d1 = i-1; d2 = j-1;
        I1 = find(trnlbls == d1); I2 = find(trnlbls == d2);
        D1 = Utrn_r'*trndat(:,I1); D2 = Utrn_r'*trndat(:,I2);
        train = [D1 D2];
        group = [trnlbls(I1); trnlbls(I2)];
        Md1=fitcsvm(train',group);
        T1 = find(testlbls==d1); T2 = find(testlbls==d2);
        test = [Utrn_r*testdat(:,T1) Utrn_r*testdat(:,T2)];
        label = predict(Md1, test');
        err_svm = abs(label - [testlbls(T1); testlbls(T2)]);
        success_svm(count) = length(find(err_svm==0))/length(err_svm);
        count = count+1;
    end
end
%% svm accuracy 
[m1 m1i] = max(success_svm); [m2 m2i] = min(success_svm);
figure()
plot(success_svm,'-o'); ylim([0, 1]);
hold on
plot(m1i, m1,'ro')
hold on 
plot(m2i, m2,'go')
xticks(1:45)
xtickangle(90)
xticklabels(pairs)
xlabel('pairs'); ylabel('accuracy %')
title('SVM success rate for all digit pairs')
%%
pairs = ["(0,1)", "(0,2)", "(0,3)", "(0,4)", "(0,5)", "(0,6)", "(0,7)", "(0,8)", "(0,9)", "(1,2)", "(1,3)", "(1,4)", "(1,5)", "(1,6)", "(1,7)", "(1,8)", "(1,9)", "(2,3)", "(2,4)", "(2,5)", "(2,6)", "(2,7)", "(2,8)", "(2,9)", "(3,4)", "(3,5)", "(3,6)", "(3,7)", "(3,8)", "(3,9)", "(4,5)", "(4,6)", "(4,7)", "(4,8)", "(4,9)", "(5,6)", "(5,7)", "(5,8)", "(5,9)", "(6,7)", "(6,8)", "(6,9)", "(7,8)", "(7,9)", "(8,9)"];