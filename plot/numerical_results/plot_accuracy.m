function plot_accuracy(gpu)

arguments (Input)
    gpu (1,1) string = "RTX4090Laptop"
end

FontSize = 8;
type_in = "d";

%% get data
dir_name = dir(gpu + "/oz2_results_" + type_in + "gemm_accuracy*");
file_name = gpu + "/" + dir_name.name;
data = detectImportOptions(file_name);

idx = 1;
data.SelectedVariableNames = idx;
data_phi = readmatrix(file_name,data);
data_phi = data_phi(2:end,:);

idx = 4;
data.SelectedVariableNames = idx;
data_k = readmatrix(file_name,data);
data_k = data_k(2:end,:);

idx = 5;
data.SelectedVariableNames = idx;
data_func = readmatrix(file_name,data);
data_func = data_func(2:end,:);

data.SelectedVariableNames = idx+1:length(data.VariableNames);
data_err = readmatrix(file_name,data);
data_moduli = data_err(1,:);
data_err = data_err(2:end,:);

%% plot
yl_min = inf;
yl_max = 0;
phi = [-1, 1, 2, 4];
fig = figure('Position',[50,50,550,300]);
t = tiledlayout(1,4);
for tid = 1:4
    nexttile; hold on; grid on;

    for k = [1024, max(data_k)]

        idx = contains(data_func,"DGEMM") & data_k == k & data_phi == phi(tid);
        err = data_err(idx,:);
        plot(data_moduli, err, mark(3-2*(k==1024),1,1), 'DisplayName', "native FP64 DGEMM {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);

        idx = contains(data_func,"OS1-7") & data_k == k & data_phi == phi(tid);
        err = data_err(idx,:);
        plot(data_moduli, err, mark(3-2*(k==1024),1,2), 'DisplayName', "INT8-based Ozaki-I (7 slices) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);

        idx = contains(data_func,"OS2-i8-fast") & data_k == k & data_phi == phi(tid);
        err = data_err(idx,:);
        plot(data_moduli, err, mark(3-2*(k==1024),1,3), 'DisplayName', "INT8 Ozaki-II (fast) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);

        idx = contains(data_func,"OS2-i8-accu") & data_k == k & data_phi == phi(tid);
        err = data_err(idx,:);
        plot(data_moduli, err, mark(3-2*(k==1024),1,4), 'DisplayName', "INT8 Ozaki-II (acc.) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);

        idx = contains(data_func,"OS2-f8-fast") & data_k == k & data_phi == phi(tid);
        err = data_err(idx,:);
        plot(data_moduli, err, mark(3-2*(k==1024),1,5), 'DisplayName', "FP8 Ozaki-II (fast) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);

        idx = contains(data_func,"OS2-f8-accu") & data_k == k & data_phi == phi(tid);
        err = data_err(idx,:);
        plot(data_moduli, err, mark(3-2*(k==1024),1,6), 'DisplayName', "FP8 Ozaki-II (acc.) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);

        xlim([min(data_moduli), max(data_moduli)]);
        xticks(data_moduli);
    end

    if phi(tid)<0
        title("Std. normal", 'Interpreter','tex');
    else
        title("{\it\phi = " + phi(tid) + "}", 'Interpreter','tex');
    end
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold','YScale','Log');
    ylim('padded');
    yl_tmp = ylim;
    yl_min = min(yl_min, yl_tmp(1));
    yl_max = max(yl_max, yl_tmp(2));
end

for tid = 1:4
    nexttile(tid);
    ylim([yl_min, yl_max]);
    yticks(10.^(-20:4:20));
    if tid > 1
        yticklabels([]);
    end
end

lgd = legend('Interpreter','tex','FontName','Yu Gothic UI Semibold','IconColumnWidth',15,'NumColumns',2);
lgd.Layout.Tile = 'north';
t.TileSpacing = "tight";
t.Padding = "compact";
xlabel(t,"Number of moduli",'FontName','Yu Gothic UI Semibold');
ylabel(t,"Max. relative error",'FontName','Yu Gothic UI Semibold');
set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');

savefig(fig,gpu+"/"+gpu+"_accuracy_"+type_in);
exportgraphics(fig,gpu+"/"+gpu+"_accuracy_"+type_in+".png",'Resolution',600);
end

%%
function m = mark(i,j,k)
lines = {"-",":","--",""};
markers = {"", "o", "x", "d", "p", "+", ".", "s", "h", "^", "v", ">", "<"};
colors = {"k", "c", "r", "b", "g", "m"};
m = lines{i} + markers{j} + colors(k);
end
