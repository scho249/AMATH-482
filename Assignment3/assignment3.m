clear; close all; clc; 
%% Part 1: Ideal

load('cam1_1.mat') 
load('cam2_1.mat')
load('cam3_1.mat')
% implay(vidFrames3_1)

nF1_1 = size(vidFrames1_1,4);
nF2_1 = size(vidFrames2_1,4);
nF3_1 = size(vidFrames3_1,4);

for j = 1:nF1_1
    cam1(j).cdata = vidFrames1_1(:,:,:,j);
    cam1(j).colormap = [];
end

for j = 1:nF2_1
    cam2(j).cdata = vidFrames2_1(:,:,:,j);
    cam2(j).colormap = [];
end

for j = 1:nF3_1
    cam3(j).cdata = vidFrames3_1(:,:,:,j);
    cam3(j).colormap = [];
end

%% Change to grayscale
% cam1
pos1 = [];
for j = 1:nF1_1
    X1gr = rgb2gray(frame2im(cam1(j)));
    %filter can
     X1gr(:,1:310) = 0;
     X1gr(:,390:end) = 0;
     X1gr(1:200,:) = 0;
     X1gr(400:end,:) = 0;
    [mx, ind] = max(X1gr(:));
    [x1, y1] = ind2sub(size(X1gr), ind);
     pos1 = [pos1; mean(x1), mean(y1)];
end

% cam2
pos2 = [];
for j = 1:nF2_1
    X2gr = rgb2gray(frame2im(cam2(j)));
    %filter can
    X2gr(:,1:250) = 0;
    X2gr(:,330:end) = 0;
    X2gr(1:90,:) = 0;
    X2gr(380:end,:) = 0;
    [mx, ind] = max(X2gr(:));
    [x2, y2] = ind2sub(size(X2gr), ind);
     pos2 = [pos2; x2, y2];
end

% cam 3
pos3 = [];
for j = 1:nF3_1
    X3gr = rgb2gray(frame2im(cam3(j)));
    %filter can
    X3gr(:,1:260) = 0;
    X3gr(:,490:end) = 0;
    X3gr(1:240,:) = 0;
    X3gr(330:end,:) = 0;
     [mx, ind] = max(X3gr(:));
    [x3, y3] = ind2sub(size(X3gr), ind);
     pos3 = [pos3; mean(x3), mean(y3)];
end

%% trimming
[m ind] = min(pos1(1:20,2)); 
pos1 = pos1(ind:end,:);
[m ind] = min(pos2(1:20,2));
pos2 = pos2(ind:end,:);
[m ind] = min(pos3(1:20,2));
pos3 = pos3(ind:end,:);
pos2 = pos2(1:length(pos3),:); pos1 = pos1(1:length(pos3),:);

% data matrix X
X = [pos1'; pos2'; pos3'];
n = length(pos1); ave = mean(X,2);
X = X-repmat(ave,1,n);
[U,S,V] = svd(X, 'econ');
sig = diag(S);

%% plotting
figure()
subplot(3,1,1)
plot(sig, 'ko','Linewidth',1)
ylabel('\sigma')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',1)
ylabel('Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,3)
plot(cumsum(sig.^2)/sum(sig.^2),'ko','Linewidth',1)
ylabel('Cumulative Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)

figure()
plot(V(:,1), 'LineWidth', .8)
hold on
plot(V(:,2), 'LineWidth', .8)
xlim([0 225])
xlabel('t');  legend('mode 1', 'mode 2');


%%
clear; close all; clc;
%% Case 2: Noisy 
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')

nF1_2 = size(vidFrames1_2,4);
nF2_2 = size(vidFrames2_2,4); % max size
nF3_2 = size(vidFrames3_2,4);

for j = 1:nF2_2
    if j <= nF1_2
        cam1(j).cdata = vidFrames1_2(:,:,:,j);
        cam1(j).colormap = [];
    end
    if j <= nF3_2
        cam3(j).cdata = vidFrames3_2(:,:,:,j);
        cam3(j).colormap = [];
    end
    cam2(j).cdata = vidFrames2_2(:,:,:,j);
    cam2(j).colormap = [];
end

%%
pos1 = []; pos2 = []; pos3 = [];
for j = 1:nF2_2
    if j <= nF1_2
        X1gr = rgb2gray(frame2im(cam1(j)));
        %filter can
        X1gr(:,1:324) = 0;
        X1gr(:,460:end) = 0;
        X1gr(1:180,:) = 0;
        X1gr(370:end,:) = 0;
        [mx, ind] = max(X1gr(:));
        [x1, y1] = ind2sub(size(X1gr), ind);
        pos1 = [pos1; x1, y1];
    end
    if j <= nF3_2
        X3gr = rgb2gray(frame2im(cam3(j)));
        %filter can
        X3gr(:,1:240) = 0;
        X3gr(:,540:end) = 0;
        X3gr(1:180,:) = 0;
        X3gr(380:end,:) = 0;
        [mx, ind] = max(X3gr(:));
        [x3, y3] = ind2sub(size(X3gr), ind);
        pos3 = [pos3; x3, y3];
    end
    X2gr = rgb2gray(frame2im(cam2(j)));
    %filter can
    X2gr(:,1:420) = 0;
    X2gr(:,380:end) = 0;
    X2gr(1:80,:) = 0;
    X2gr(400:end,:) = 0;
    [mx, ind] = max(X2gr(:));
    [x2, y2] = ind2sub(size(X2gr), ind);
     pos2 = [pos2; x2, y2];
end
%%
[m ind] = min(pos1(1:20,2)); 
pos1 = pos1(ind:end,:);
[m ind] = min(pos2(1:20,2));
pos2 = pos2(ind:end,:);
[m ind] = min(pos3(1:20,2));
pos3 = pos3(ind:end,:);
pos2 = pos2(1:length(pos1), :); pos3 = pos3(1:length(pos1), :);

X = [pos1'; pos2'; pos3'];
n = length(pos1);
ave = mean(X,2);
X = X-repmat(ave,1,n);
[U,S,V] = svd(X, 'econ');
sig = diag(S);

%% plotting
figure()
subplot(3,1,1)
plot(sig, 'ko','Linewidth',1)
ylabel('\sigma')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',1)
ylabel('Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,3)
plot(cumsum(sig.^2)/sum(sig.^2),'ko','Linewidth',1)
ylabel('Cumulative Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)

figure()
plot(V(:,1), 'LineWidth', .5)
hold on
plot(V(:,2), 'LineWidth', .5)
hold on
plot(V(:,3), 'LineWidth', .5)
axis([0 312 -0.18 0.18])
xlabel('t');  legend('mode 1', 'mode 2', 'mode 3');


%%
clear; close all; clc; 
%% Part 3: Horizontal Displacement
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')

nF1_3 = size(vidFrames1_3,4);
nF2_3 = size(vidFrames2_3,4); % max size
nF3_3 = size(vidFrames3_3,4);

for j = 1:nF2_3
    if j <= nF1_3
        cam1(j).cdata = vidFrames1_3(:,:,:,j);
        cam1(j).colormap = [];
    end
    if j <= nF3_3
        cam3(j).cdata = vidFrames3_3(:,:,:,j);
        cam3(j).colormap = [];
    end
    cam2(j).cdata = vidFrames2_3(:,:,:,j);
    cam2(j).colormap = [];
end

pos1 = []; pos2 = []; pos3 = [];
for j = 1:nF2_3
    if j <= nF1_3
        X1gr = rgb2gray(frame2im(cam1(j)));
        %filter can
        X1gr(:,1:250) = 0;
        X1gr(:,410:end) = 0;
        X1gr(1:200,:) = 0;
        X1gr(400:end,:) = 0;
        [mx, ind] = max(X1gr(:));
        [x1, y1] = ind2sub(size(X1gr), ind);
        pos1 = [pos1; x1, y1];
    end
    if j <= nF3_3
        X3gr = rgb2gray(frame2im(cam3(j)));
        %filter can
        X3gr(:,1:270) = 0;
        X3gr(:,490:end) = 0;
        X3gr(1:150,:) = 0;
        X3gr(330:end,:) = 0;
        [mx, ind] = max(X3gr(:));
        [x3, y3] = ind2sub(size(X3gr), ind);
        pos3 = [pos3; x3, y3];
    end
    X2gr = rgb2gray(frame2im(cam2(j)));
    %filter can
    X2gr(:,1:200) = 0;
    X2gr(:,400:end) = 0;
    X2gr(1:150,:) = 0;
    X2gr(400:end,:) = 0;
    [mx, ind] = max(X2gr(:));
    [x2, y2] = ind2sub(size(X2gr), ind);
     pos2 = [pos2; x2, y2];
end

%%
[m ind] = min(pos1(1:20,2)); 
pos1 = pos1(ind:end,:);
[m ind] = min(pos2(1:20,2));
pos2 = pos2(ind:end,:);
[m ind] = min(pos3(1:20,2));
pos3 = pos3(ind:end,:);
pos1 = pos1(1:length(pos3),:); pos2 = pos2(1:length(pos3),:);

X = [pos1'; pos2'; pos3'];
n = length(pos1);
ave = mean(X,2);
X = X-repmat(ave,1,n);
[U,S,V] = svd(X, 'econ');
sig = diag(S);

%% plotting
figure()
subplot(3,1,1)
plot(sig, 'ko','Linewidth',1)
ylabel('\sigma')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',1)
ylabel('Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,3)
plot(cumsum(sig.^2)/sum(sig.^2),'ko','Linewidth',1)
ylabel('Cumulative Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)

figure()
plot(V(:,1), 'LineWidth', .8)
hold on
plot(V(:,2), 'LineWidth', .8)
hold on
plot(V(:,3), 'LineWidth', .8)
xlim([0 201])
xlabel('t');  legend('mode 1', 'mode 2', 'mode 3');


%%
clear; close all; clc;
%% Part 4: Horizontal Displacement and Rotation
load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')

nF1_4 = size(vidFrames1_4,4);
nF2_4 = size(vidFrames2_4,4); % max size
nF3_4 = size(vidFrames3_4,4);

for j = 1:nF2_4
    if j <= nF1_4
        cam1(j).cdata = vidFrames1_4(:,:,:,j);
        cam1(j).colormap = [];
    end
    if j <= nF3_4
        cam3(j).cdata = vidFrames3_4(:,:,:,j);
        cam3(j).colormap = [];
    end
    cam2(j).cdata = vidFrames2_4(:,:,:,j);
    cam2(j).colormap = [];
end
%%
pos1 = []; pos2 = []; pos3 = [];
for j = 1:nF2_4
    if j <= nF1_4
        X1gr = rgb2gray(frame2im(cam1(j)));
        %filter can
        X1gr(:,1:320) = 0;
        X1gr(:,450:end) = 0;
        X1gr(1:230,:) = 0;
        X1gr(380:end,:) = 0;
        [m1, ind1] = max(X1gr(:));
        [x1, y1] = ind2sub(size(X1gr), ind1);
        pos1 = [pos1; x1, y1];
    end
    if j <= nF3_4
        X3gr = rgb2gray(frame2im(cam3(j)));
        %filter can
        X3gr(:,1:300) = 0;
        X3gr(:,470:end) = 0;
        X3gr(1:160,:) = 0;
        X3gr(270:end,:) = 0;
        [m3, ind3] = max(X3gr(:));
        [x3, y3] = ind2sub(size(X3gr), ind3);
        pos3 = [pos3; x3, y3];
    end
    X2gr = rgb2gray(frame2im(cam2(j)));
    %filter can
    X2gr(:,1:230) = 0;
    X2gr(:,410:end) = 0;
    X2gr(1:95,:) = 0;
    X2gr(330:end,:) = 0;
    [m2, ind2] = max(X2gr(:));
    [x2, y2] = ind2sub(size(X2gr), ind2);
     pos2 = [pos2; x2, y2];
end

%%
[m ind] = min(pos1(1:20,2)); 
pos1 = pos1(ind:end,:);
[m ind] = min(pos2(1:20,2));
pos2 = pos2(ind:end,:);
[m ind] = min(pos3(1:20,2));
pos3 = pos3(ind:end,:);
pos2 = pos2(1:length(pos3),:); pos1 = pos1(1:length(pos3),:);

X = [pos1'; pos2'; pos3'];
n = length(pos1);
ave = mean(X,2);
X = X-repmat(ave,1,n);
[U,S,V] = svd(X, 'econ');
sig = diag(S);

%% plotting
figure()
subplot(3,1,1)
plot(sig, 'ko','Linewidth',1)
ylabel('\sigma')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',1)
ylabel('Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)
subplot(3,1,3)
plot(cumsum(sig.^2)/sum(sig.^2),'ko','Linewidth',1)
ylabel('Cumulative Energy')
set(gca,'Fontsize',12,'Xtick',0:1:6)

figure()
plot(V(:,1), 'LineWidth', .5)
hold on
plot(V(:,2), 'LineWidth', .5)
hold on
plot(V(:,3), 'LineWidth', .5)
hold on 
plot(V(:,4), 'c', 'LineWidth', .5)
axis([0 383 -0.25 0.25])
xlabel('t');  legend('mode 1', 'mode 2', 'mode 3', 'mode 4');
