clear; close all; clc;

%% part 1: GNR

figure(1) % GNR
[y_g, Fs_g] = audioread('GNR.m4a'); % y = sampled data; Fs = sample rate for y
y_g = y_g.';
y_g = y_g(1:length(y_g));
tr_gnr = length(y_g)/Fs_g; % record time in seconds
plot((1:length(y_g))/Fs_g,y_g);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O'' Mine');
%p8 = audioplayer(y,Fs); playblocking(p8);

L = tr_gnr; n = length(y_g); 
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];ks = fftshift(k); %scaling by 1/L to get unit Hz instead of angular freq

%% Gabor transform

tau = 0:0.2:13;
a = 200;
gnr_spec = [];
for j = 1:length(tau)
    gabor = exp(-a*(t - tau(j)).^2);
    yg_g = gabor.*y_g;
    ygt_g = fft(yg_g);
    % filtering overtones
    [maxfreq(j), maxfreqind(j)] = max(abs(ygt_g));
    ygtf_g = ygt_g.*exp(-0.05*(k-k(maxfreqind(j))).^2);
    gnr_spec = [gnr_spec; abs(fftshift(ygtf_g))];
end

%% plotting example gabor transform 

gabor = exp(-a*(t - 4).^2);
yg_g = gabor.*y_g;
figure(2)
subplot(3,1,1)
plot(t,y_g), hold on
plot(t,gabor, 'r');
xlabel('time (t)'), ylabel('y_g(t)')
subplot(3,1,2)
plot(t,yg_g)
xlabel('time (t)'), ylabel('y_g(t)*g(t-\tau)')
subplot(3,1,3)
plot(ks,abs(fftshift(fft(yg_g)))/max(abs(fft(yg_g))))
xlabel('frequency (k)'), ylabel('fft(y(t)*g(t-\tau))')

%% GNR spectrogram

figure(3)
pcolor(tau,ks,log10(abs(gnr_spec.')+1)), shading interp
colormap(hot)
axis([0 13 0 1000]);
% musical note
yticks([277.18 312 370 415.3 554.36 698.46 739.99]);
yticklabels({'C#4/Db4', 'D#4', 'F#4/Gb4', 'G#4/Ab4', 'C#5/Db5', 'F5' 'F#5/Gb5'});
colorbar;
ax = gca
ax.YGrid = 'on'
ax.Layer = 'top'
ax.GridAlpha = 0.5;
ax.GridColor = 'w';
%xlabel('time (t)'), ylabel('frequency (k)')
xlabel('time (t)'), ylabel('music note')
title(['a = ',num2str(a)],'Fontsize',16)

%% part 2: Floyd bass (first 15 seconds)
clear; close all; clc;

figure(3) % floyd audio plot
[y_f, Fs_f] = audioread('Floyd.m4a'); % y = sampled data; Fs = sample rate for y
y_f = y_f.'; 
 % record time in seconds
plot((1:length(y_f))/Fs_f,y_f);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb');
%p8 = audioplayer(y_f,Fs_f); playblocking(p8);

% Gabor transform
y_f = y_f(1:700000);
tr_floyd = length(y_f)/Fs_f;
L = tr_floyd; n = length(y_f); 
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];ks = fftshift(k);
a = 10;
tau = 0:0.2:L;
floyd_spec = [];

%% (next 15 seconds)
clear; close all; clc; 
[y_f, Fs_f] = audioread('Floyd.m4a'); % y = sampled data; Fs = sample rate for y
y_f = y_f.'; 
y_f = y_f(700001:1300000);
tr_floyd = length(y_f)/Fs_f; % record time in seconds
plot((700001:700000+length(y_f))/Fs_f,y_f);

% Gabor transform
L = tr_floyd; n = length(y_f); 
t2 = linspace((700001/Fs_f),(700001/Fs_f)+L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];ks = fftshift(k);
a = 10;
tau = (700001/Fs_f):0.1:(700001/Fs_f)+L;
floyd_spec = [];

%% filtering bass using Shannon filter

yt_f = fft(y_f);
shannon = abs(k) >= 200;
yt_f(shannon) = 0;
yif_f = ifft(yt_f);

%% Gabor transform

for j = 1:length(tau)
    gabor = exp(-a*(t - tau(j)).^2);
    yg_f = gabor.*yif_f;
    ygt_f = fft(yg_f);
    % filtering overtones
    [bassfreq, bassfreqind(j)] = max(abs(ygt_f));
    ygtf_f = ygt_f.*exp(-0.05*(k-k(bassfreqind(j))).^2);
    floyd_spec = [floyd_spec; abs(fftshift(ygtf_f))];
end
%% sample diagram of gabor transform
gabor = exp(-a*(t - 5).^2);
yg_f = gabor.*y_f;

subplot(3,1,1)
plot(t,y_f), hold on
plot(t,gabor, 'r');
xlabel('time (t)'), ylabel('y_f(t)')
subplot(3,1,2)
plot(t,yg_f)
xlabel('time (t)'), ylabel('y_f(t)*g(t-\tau)')
subplot(3,1,3)
plot(ks,abs(fftshift(fft(yg_f)))/max(abs(fft(yg_f))))
xlabel('frequency (k)'), ylabel('fft(y(t)*g(t-\tau))')

%% Floyd bass spectrogram
% need to change the axis according to each part
figure(4)
pcolor(tau,ks,floyd_spec.'), shading interp
colormap(hot)
%axis([0 15.9 40 150]);
axis([15.9 28.8 40 150]); % part 2 axis
%music note
yticks([82.41 92.5 98 110 123.47]);
yticklabels({'E2', 'F#2/Gb2', 'G2', 'A2', 'B2'});
colorbar
ax = gca
ax.YGrid = 'on'
ax.Layer = 'top'
ax.GridAlpha = 0.5;
ax.GridColor = 'w';
%xlabel('time (t)'), ylabel('frequency (k)')
xlabel('time (t)'), ylabel('music note')
title(['a = ',num2str(a)],'Fontsize',16)

%%
plot(t, yif_f)
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb bass');
p8 = audioplayer(yif_f,Fs_f); playblocking(p8);
audiowrite('floydbass.m4a',yif_f,Fs_f);
%% part 3: Floyd guitar (first 10 seconds)
clear; close all; clc;
% filtering guitar using Shannon filter
[y_f, Fs_f] = audioread('Floyd.m4a'); % y = sampled data; Fs = sample rate for y
y_f = y_f.'; 
y_f = y_f(1:480000);
tr_floyd = length(y_f)/Fs_f; % record time in seconds
%p8 = audioplayer(y_f,Fs_f); playblocking(p8);
% Gabor transform
L = tr_floyd; n = length(y_f); 
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];ks = fftshift(k);
a = 100;
tau = 0:0.3:L;
guitar_spec = [];

yt_f = fft(y_f);

% shannon filter
guitar1 = k <= 450; 
guitar2 = k >= 1000;
yt_f(guitar1) = 0; yt_f(guitar2) = 0;
yif_f = ifft(yt_f);
%% Gabor transform
for j = 1:length(tau)
    gabor = exp(-a*(t - tau(j)).^2);
    yg_f = gabor.*yif_f;
    ygt_f = fft(yg_f);
    % filtering overtones
    [guitarfreq, guitarfreqind(j)] = max(abs(ygt_f));
    ygtf_f = ygt_f.*exp(-0.05*(k-k(guitarfreqind(j))).^2);
    guitar_spec = [guitar_spec; abs(fftshift(ygtf_f))];
end
%% Guitar Spectrogram
pcolor(tau,ks,log10(abs(guitar_spec.')+1)), shading interp
colormap(hot)
axis([0 L 450 1100]);
% music note
yticks([493.88 587.33 659.25 739.99 783.99 880 987.77]);
yticklabels({'B4','D5', 'E5' 'F#5/Gb5', 'G5', 'A5', 'B5'});
colorbar
ax = gca
ax.YGrid = 'on'
ax.Layer = 'top'
ax.GridAlpha = 0.5;
ax.GridColor = 'w';
%xlabel('time (t)'), ylabel('frequency (k)')
xlabel('time (t)'), ylabel('music note')
title(['a = ',num2str(a)],'Fontsize',16)

