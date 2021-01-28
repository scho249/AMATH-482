clear all; close all; clc;

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
[X,Y,Z]=meshgrid(x,y,z); % real-space grid vectors
[Kx,Ky,Kz]=meshgrid(ks,ks,ks); % frequency-space grid vectors


% Averaging 
ave = zeros(n,n,n);
for j=1:49
  Un(:,:,:)=reshape(subdata(:,j),n,n,n);
  M = max(abs(Un),[],'all');
  Unt = fftn(Un);
  ave = ave + Unt;
end

ave = abs(fftshift(ave))./49; % averaged signal
[max_ave, mave_ind] = max(ave(:)); 
[mave_x, mave_y, mave_z] = ind2sub(size(ave), mave_ind); % center frequency
                                                         % coordinates

% figure(1) % visualize signal in real-space
isosurface(X,Y,Z,abs(Un),0.4)
axis([-L L -L L -L L]), grid on, drawnow
xlabel('X'), ylabel('Y'), zlabel('Z')
% 
figure(2) % visualize averaged signal in frequency-space
isosurface(Kx,Ky,Kz,abs(ave)/max_ave,0.7), grid on
xlabel('Kx'), ylabel('Ky'), zlabel('Kz')


% Gaussian filter
tau = 0.02; % Width of filter
Kxi = Kx(mave_x, mave_y, mave_z); %
Kyi = Ky(mave_x, mave_y, mave_z); % Center of the filter
Kzi = Kz(mave_x, mave_y, mave_z); % 

% Define the filter
filter = exp(-tau*(((Kx - Kxi).^2) + ((Ky - Kyi).^2) + ((Kz - Kzi).^2)));

path_X = []; path_Y = []; path_Z = [];
for jj=1:49
    Un(:,:,:)=reshape(subdata(:,jj),n,n,n);
    Unt = fftn(Un);
    Unft = filter .*  fftshift(Unt); % applying filter
    Unf = ifftn(Unft); % real-space data
    
    [max_val, max_val_ind] = max(abs(Unf(:))); 
    [max_x, max_y, max_z] = ind2sub(size(Unf), max_val_ind);
    % real-space coordinates of submarine position
    path_X(jj) = X(max_x, max_y, max_z);
    path_Y(jj) = Y(max_x, max_y, max_z);
    path_Z(jj) = Z(max_x, max_y, max_z);
    
    figure(3) % visualize trajectory (3D)
    isosurface(X, Y, Z, abs(Unf)/max_val, 0.8)
    axis([-L, L, -L, L, -L, L]), grid on, drawnow
    xlabel('X'), ylabel('Y'), zlabel('Z')
end
set(patch(isosurface(X, Y, Z, abs(Unf)/max_val, 0.8)), 'FaceColor', 'm', 'EdgeColor', 'none');
final_position = [path_X(49), path_Y(49)] % current position 

figure(4) % visualize trajectory 
plot3(path_X, path_Y, path_Z, 'Linewidth', 1.5)
axis([-L, L, -L, L, -L, L]), grid on, hold on 
plot3(path_X(49), path_Y(49), path_Z(49), 'm.', 'Markersize', 20)
xlabel('X'), ylabel('Y'), zlabel('Z')

