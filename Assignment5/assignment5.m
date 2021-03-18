clear; close all; clc;
%% vid 1: monte_carlo
mc_vid = VideoReader('monte_carlo_low.mp4');
mc_fr = read(mc_vid);
numFr = get(mc_vid,'numberOfFrames'); runtime = get(mc_vid,'Duration');

%% convert to grayscale -> video matrx SVD

for i=1:numFr
    mc_gray = rgb2gray(mc_fr(:,:,:,i));
    mc_mat(:,i) = double(reshape(mc_gray, [], 1));
end
dt = runtime/numFr;
%%
% SVD 
X1 = mc_mat(:,1:end-1); X2 = mc_mat(:, 2:end);
[U, Sig, V] = svd(X1, 'econ');

%% plot singular values

plot(diag(Sig), 'ko')
ylabel('\sigma_i')

%%
mode = 3;
U = U(:,1:mode); Sig = Sig(1:mode, 1:mode); V = V(:,1:mode);
S = U'*X2*V/Sig;

%% eigendecomp
[eV, D] = eig(S);
mu = diag(D);
omega = log(mu)/dt;

%% plot omega
line = 15:15;
plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
hold on
plot(real(omega)*dt, imag(omega)*dt, 'bo', 'Markersize' ,5)
hold on
yline(0); 
hold on
xline(0);
xlim([-1 0.1]); ylim([-0.6 0.6]);
xlabel('Re(\omega)')
ylabel('Im(\omega)')

%% separate foreground and background
thresh = 0.01;
bg_ind = find(abs(omega) < thresh);
omega = omega(bg_ind);
Phi = U*eV;
Phi = Phi(:,bg_ind);
y0 = Phi\X1(:,1);
%%
sz = size(X1,2);
t = dt*(0:sz-1);

u_modes = zeros(length(y0), length(t));
for i=1:length(t)-1
    u_modes(:,i) = y0.*exp(omega*t(i));
end
Xdmd = Phi*u_modes; % background video 

%% sparse, background, foreground matrix
Xsp = X1 - abs(Xdmd);
R = Xsp.*(Xsp<0);
Xbg = R + abs(Xdmd);
Xfg = Xsp - R;

Xre = Xbg + Xfg;

%% plot

figure()
subplot(4,3,1)
imshow(uint8(reshape(X1(:,50), 540,[])))
subplot(4,3,2)
imshow(uint8(reshape(X1(:,65), 540,[])))
title('Original Video', 'FontSize', 11)
subplot(4,3,3)
imshow(uint8(reshape(X1(:,80), 540,[])))
subplot(4,3,4)
imshow(uint8(reshape(Xdmd(:,50), 540,[])))
subplot(4,3,5)
imshow(uint8(reshape(Xdmd(:,65), 540,[])))
title('Background (residual not added)', 'FontSize', 11)
subplot(4,3,6)
imshow(uint8(reshape(Xdmd(:,80), 540,[])))
subplot(4,3,7)
imshow(uint8(reshape(Xfg(:,50), 540,[])))
subplot(4,3,8)
imshow(uint8(reshape(Xfg(:,65), 540,[])))
title('Foreground (residual subtracted)', 'FontSize', 11)
subplot(4,3,9)
imshow(uint8(reshape(Xfg(:,80), 540,[])))
subplot(4,3,10)
imshow(uint8(reshape(Xre(:,50), 540,[])))
subplot(4,3,11)
imshow(uint8(reshape(Xre(:,65), 540,[])))
title('Reconstructed Video', 'FontSize', 11)
subplot(4,3,12)
imshow(uint8(reshape(Xre(:,80), 540,[])))

%%
clear; close all; clc; 
%% vid 2: ski_drop
ski_vid = VideoReader('ski_drop_low.mp4');
ski_fr = read(ski_vid);
numFr = get(ski_vid,'numberOfFrames'); runtime = get(ski_vid,'Duration');

% convert to grayscale -> video matrx SVD
cropsz = [520 400]; % crop size
for i=1:numFr
    ski_gray = rgb2gray(ski_fr(:,:,:,i));
    window = centerCropWindow2d(size(ski_gray), cropsz);
    ski_gray = imcrop(ski_gray,window);
    %imshow(ski_gray)
    ski_mat(:,i) = double(reshape(ski_gray, [], 1));
end
dt = runtime/numFr;

% SVD 
X1 = ski_mat(:,1:end-1); X2 = ski_mat(:, 2:end);
[U, Sig, V] = svd(X1, 'econ');
mode = 20;

%% plot singular values

plot(diag(Sig), 'ko')
ylabel('\sigma_i')

%%
U = U(:,1:mode); Sig = Sig(1:mode, 1:mode); V = V(:,1:mode);
S = U'*X2*V*diag(1./diag(Sig));

% eigendecomp
[eV, D] = eig(S);
mu = diag(D);
omega = log(mu)/dt;

%% plot omega
line = 15:15;
plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
hold on
plot(real(omega)*dt, imag(omega)*dt, 'bo', 'Markersize' ,5)
hold on
yline(0); 
hold on
xline(0);
xlim([-1 0.1]); ylim([-0.6 0.6]);
xlabel('Re(\omega)')
ylabel('Im(\omega)')

%% separate foreground and background
thresh = 0.01;
bg_ind = find(abs(omega) < thresh);
omega = omega(bg_ind);
Phi = U*eV;
Phi = Phi(:,bg_ind);
y0 = Phi\X1(:,1);
%%
sz = size(X1,2);
t = dt*(0:sz-1);

u_modes = zeros(length(y0), length(t));
for i=1:length(t)-1
    u_modes(:,i) = y0.*exp(omega*t(i));
end
Xdmd = Phi*u_modes; % background video 

%% sparse, background, foreground matrix
Xsp = X1 - abs(Xdmd);
R = Xsp.*(Xsp<0);
Xbg = R + abs(Xdmd);
Xfg = Xsp - R;
Xre = Xbg + Xfg;
Xfg = Xfg - R;
%% plot

figure()
subplot(4,3,1)
imshow(uint8(reshape(ski_mat(:,100), 520,[])))
subplot(4,3,2)
imshow(uint8(reshape(ski_mat(:,200), 520,[])))
title('Original Video', 'FontSize', 11)
subplot(4,3,3)
imshow(uint8(reshape(ski_mat(:,300), 520,[])))
subplot(4,3,4)
imshow(uint8(reshape(Xdmd(:,100), 520,[])))
subplot(4,3,5)
imshow(uint8(reshape(Xdmd(:,200), 520,[])))
title('Background (residual not added)', 'FontSize', 11)
subplot(4,3,6)
imshow(uint8(reshape(Xdmd(:,300), 520,[])))
subplot(4,3,7)
imshow(uint8(reshape(Xfg(:,100), 520,[])))
subplot(4,3,8)
imshow(uint8(reshape(Xfg(:,200), 520,[])))
title('Foreground (residual subtracted)', 'FontSize', 11)
subplot(4,3,9)
imshow(uint8(reshape(Xfg(:,300), 520,[])))
subplot(4,3,10)
imshow(uint8(reshape(Xre(:,100), 520,[])))
subplot(4,3,11)
imshow(uint8(reshape(Xre(:,200), 520,[])))
title('Reconstructed Video', 'FontSize', 11)
subplot(4,3,12)
imshow(uint8(reshape(Xre(:,300), 520,[])))
