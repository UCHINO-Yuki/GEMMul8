function plot_flops_multArch

close all
clc

GPUs = ["RTX4090Laptop" "RTX5080" "RX9070XT" "GB10" "GB200"];
GPUNAME = ["NVIDIA RTX 4090 Laptop" "NVIDIA RTX 5080" "AMD RX 9070 XT" "NVIDIA GB10" "NVIDIA GB200 NVL4"];
SIZE = [1024 2048 4096 8192 16384 32768];

FontSize = 8;
LINEWIDTH = 1.2;
MARKERSIZE = 6;

fig = figure('Position',[50,50,550,400]);
tile = tiledlayout(2,3);
yl = cell(3,1);
yl{1} = [inf,0]; yl{2}=yl{1}; yl{3}=yl{1};

for gpu = GPUs

    dir_name = dir(gpu + "/oz2_results_dgemm_time_*");
    file_name = gpu + "/" + dir_name.name;
    data = detectImportOptions(file_name);

    idx = find(strcmp(data.VariableNames,"m"));
    data.SelectedVariableNames = idx;
    data_m = readmatrix(file_name,data);

    idx = find(strcmp(data.VariableNames,"k"));
    data.SelectedVariableNames = idx;
    data_k = readmatrix(file_name,data);

    idx = find(contains(data.VariableNames,"unction"));
    data.SelectedVariableNames = idx;
    data_func = readmatrix(file_name,data);

    idx = find(contains(data.VariableNames,"TFLOPS"));
    data.SelectedVariableNames = idx;
    data_tflops = readmatrix(file_name,data);

    tflops_64f    = nan(length(SIZE),1);
    tflops_Oz1_7  = nan(length(SIZE),1);
    tflops_Oz1_11 = nan(length(SIZE),1);
    tflops_Oz2_i15_accu = nan(length(SIZE),1);
    tflops_Oz2_i15_fast = nan(length(SIZE),1);
    tflops_Oz2_f12_accu = nan(length(SIZE),1);
    tflops_Oz2_f12_fast = nan(length(SIZE),1);

    for n = SIZE

        idx_square = data_m == n & data_k == n;

        idx = idx_square & strcmp(data_func, "DGEMM");
        if any(idx)
            tflops_64f(n == SIZE) = data_tflops(idx);
        end

        idx = idx_square & strcmp(data_func, "OS1-7");
        if any(idx)
            tflops_Oz1_7(n == SIZE) = data_tflops(idx);
        end

        idx = idx_square & strcmp(data_func, "OS1-11");
        if any(idx)
            tflops_Oz1_11(n == SIZE) = data_tflops(idx);
        end

        idx = idx_square & strcmp(data_func, "OS2-i8-accu-15");
        if any(idx)
            tflops_Oz2_i15_accu(n == SIZE) = data_tflops(idx);
        end

        idx = idx_square & strcmp(data_func, "OS2-i8-fast-15");
        if any(idx)
            tflops_Oz2_i15_fast(n == SIZE) = data_tflops(idx);
        end

        idx = idx_square & strcmp(data_func, "OS2-f8-accu-12");
        if any(idx)
            tflops_Oz2_f12_accu(n == SIZE) = data_tflops(idx);
        end

        idx = idx_square & strcmp(data_func, "OS2-f8-fast-12");
        if any(idx)
            tflops_Oz2_f12_fast(n == SIZE) = data_tflops(idx);
        end
    end

    nexttile;
    hold on;

    x = max( [nnz(isfinite(tflops_64f(:))), ...
        nnz(isfinite(tflops_Oz1_7(:))), ...
        nnz(isfinite(tflops_Oz2_i15_accu(:))), ...
        nnz(isfinite(tflops_Oz2_i15_fast(:)))] );

    plot(1:length(SIZE), tflops_64f    , mark(3,1,1), 'DisplayName', "native FP64 DGEMM",               'MarkerSize', MARKERSIZE, 'LineWidth', LINEWIDTH);
    plot(1:length(SIZE), tflops_Oz1_7  , mark(1,1,2), 'DisplayName', "INT8 Ozaki-I (7 slices)",         'MarkerSize', MARKERSIZE, 'LineWidth', LINEWIDTH);
    plot(1:length(SIZE), tflops_Oz1_11 , mark(2,1,2), 'DisplayName', "INT8 Ozaki-I (11 slices)",        'MarkerSize', MARKERSIZE, 'LineWidth', LINEWIDTH);
    plot(1:length(SIZE), tflops_Oz2_i15_fast, mark(1,3,3), 'DisplayName', "INT8 Ozaki-II fast  (15 moduli)", 'MarkerSize', MARKERSIZE, 'LineWidth', LINEWIDTH);
    plot(1:length(SIZE), tflops_Oz2_i15_accu, mark(1,4,4), 'DisplayName', "INT8 Ozaki-II accu. (15 moduli)", 'MarkerSize', MARKERSIZE, 'LineWidth', LINEWIDTH);
    plot(1:length(SIZE), tflops_Oz2_f12_fast, mark(2,3,5), 'DisplayName', "FP8 Ozaki-II fast  (12 moduli)",  'MarkerSize', MARKERSIZE, 'LineWidth', LINEWIDTH);
    plot(1:length(SIZE), tflops_Oz2_f12_accu, mark(2,4,6), 'DisplayName', "FP8 Ozaki-II accu. (12 moduli)",  'MarkerSize', MARKERSIZE, 'LineWidth', LINEWIDTH);

    xlim([1 x]);
    xticks(1:x);
    xticklabels("2^{" + log2(SIZE) + "}");
    xtickangle(0)
    ylim('padded');
    yl_tmp = ylim;
    yl{ceil(find(gpu==GPUs)/2)}(1) = min(yl{ceil(find(gpu==GPUs)/2)}(1), yl_tmp(1));
    yl{ceil(find(gpu==GPUs)/2)}(2) = max(yl{ceil(find(gpu==GPUs)/2)}(2), yl_tmp(2));


    grid on;
    set(gca,'FontName','Yu Gothic UI Semibold');
    title(GPUNAME(gpu==GPUs));
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');

end

for gpu = GPUs
    nexttile(find(gpu == GPUs));
    yl_tmp = ylim;%yl{ceil(find(gpu==GPUs)/2)};
    ylmax = ceil(yl_tmp(2)/10);
    inc = ylmax;
    if inc>1
        inc = ceil(inc/2)*2;
    end
    if inc>10
        inc = ceil(inc/10)*10;
    end
    % inc = ceil((yl_tmp(2)-yl_tmp(1))/45)*5;
    ylim([0 yl_tmp(2)]);
    yticks(0:inc:200);
end

lgd = legend('Interpreter','tex','FontName','Yu Gothic UI Semibold','IconColumnWidth',15,'NumColumns',1);
lgd.Layout.Tile = 6;
xlabel(tile,'\itm = n = k', 'Interpreter', 'tex','FontName','Yu Gothic UI Semibold');
ylabel(tile,'TFLOP/s','FontName','Yu Gothic UI Semibold');
set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
tile.TileSpacing = "tight";
tile.Padding = "compact";

savefig(fig,"flops_multArch_d");
exportgraphics(fig,"flops_multArch_d.png",'Resolution',600);


end

%%
function m = mark(i,j,k)
lines = {"-",":","--",""};
markers = {"", "o", "x", "d", "p", "+", ".", "s", "h", "^", "v", ">", "<"};
colors = {"k", "c", "r", "b", "g", "m"};
m = lines{i} + markers{j} + colors(k);
end