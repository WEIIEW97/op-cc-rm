clc, clear;


data_path = "../data/dispL_x64_U16.png";
KERNEL_RADIUS = 5;
THR = 1.5;

m = imread(data_path);
m = double(m) / 64;

figure;
imshow(m, [0, 50]);
colormap parula;
colorbar;

mp = op_kernel_cc_4n_rm(m, 5, 2.5);
figure;
imshow(mp, [0, 50]);
colormap parula;
colorbar;

