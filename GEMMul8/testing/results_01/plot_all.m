function plot_all

%% save figures
close all

mkdir fig
saveflag = true;
savedir = "fig/";
plot_accuracy(saveflag,savedir);
plot_flops(saveflag,savedir);
plot_timebreakdown(saveflag,savedir);
plot_watt(saveflag,savedir);

% %% Combining figures and generate 1 PDF
close all

imgList = dir(savedir + "oz2_results_*");
imgname = strings(length(imgList),1);
for i=1:length(imgList)
    imgname(i) = savedir + string(imgList(i).name);
end
imgname = imgname(contains(imgname,".fig"));

imgname(contains(imgname,"RTX_4090")) = [];

% rearrange
for i=1:3:length(imgname)
    tmp = imgname(i:i+2);
    RTX   = tmp(contains(tmp,"RTX"));
    A100  = tmp(contains(tmp,"A100"));
    GH200 = tmp(contains(tmp,"GH200"));
    imgname(i:i+2) = [RTX;A100;GH200];
end

% save as PDF
for i = 1:length(imgname)
    f = openfig(imgname(i));
    exportgraphics(f, savedir + "all_figures.pdf", "Resolution", 600, 'Append', i ~= 1);
end

end