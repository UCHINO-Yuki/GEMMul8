function plot_flops_multArch

close all
clc

GPUs = ["RTX4090Laptop" "RTX5080" "RX9070XT" "GH200" "GB10" "B200"];
GPUNAME = ["NVIDIA RTX 4090 Laptop" "NVIDIA RTX 5080" "AMD RX 9070 XT" "NVIDIA GH200" "NVIDIA GB10" "NVIDIA B200"];
SOFT = ["CUDA Toolkit 13.2.51", "CUDA Toolkit 13.2.51", "ROCm 7.2.1", "CUDA Toolkit 13.1.115", "CUDA Toolkit 13.0.88", "CUDA Toolkit 13.1.80"];
TYPE = ["d"];
SIZE = [1024 2048 4096 8192 16384 32768];

FontSize = 8;

for t = TYPE

    fig = figure('Position',[50,50,550,400]);
    tile = tiledlayout(2,ceil(length(GPUs)/2));
    yl = cell(3,1);
    yl{1} = [inf,0]; yl{2}=yl{1}; yl{3}=yl{1};

    for gpu = GPUs

        dir_name = dir(gpu + "/oz2_results_i_" + t + "_time*");
        flag_i8 = ~isempty(dir_name);
        if flag_i8
            file_name = gpu + "/" + dir_name.name;
            data = detectImportOptions(file_name);
            data.SelectedVariableNames = 2;
            data_i8_m = readmatrix(file_name,data);
            data.SelectedVariableNames = 4;
            data_i8_k = readmatrix(file_name,data);
            data.SelectedVariableNames = 5;
            data_i8_func = readmatrix(file_name,data);
            data.SelectedVariableNames = 6;
            data_i8_tflops = readmatrix(file_name,data);
        end

        dir_name = dir(gpu + "/oz2_results_f_" + t + "_time*");
        flag_f8 = ~isempty(dir_name);
        if flag_f8
            file_name = gpu + "/" + dir_name.name;
            data = detectImportOptions(file_name);
            data.SelectedVariableNames = 2;
            data_f8_m = readmatrix(file_name,data);
            data.SelectedVariableNames = 4;
            data_f8_k = readmatrix(file_name,data);
            data.SelectedVariableNames = 5;
            data_f8_func = readmatrix(file_name,data);
            data.SelectedVariableNames = 6;
            data_f8_tflops = readmatrix(file_name,data);
        end

        tflops_64f    = nan(length(SIZE),1);
        tflops_Oz1_7  = nan(length(SIZE),1);
        tflops_Oz2_i15 = nan(length(SIZE),1);
        tflops_Oz2_i16 = nan(length(SIZE),1);
        tflops_Oz2_f12 = nan(length(SIZE),1);
        tflops_Oz2_f13 = nan(length(SIZE),1);

        for n = SIZE
            if flag_i8
                idx = data_i8_m == n & data_i8_k == n & strcmp(data_i8_func, upper(t)+"GEMM");
                if any(idx)
                    tflops_64f(n == SIZE) = data_i8_tflops(idx);
                end

                idx = data_i8_m == n & data_i8_k == n & strcmp(data_i8_func, "Oz1-7");
                if any(idx)
                    tflops_Oz1_7(n == SIZE) = data_i8_tflops(idx);
                end

                idx = data_i8_m == n & data_i8_k == n & strcmp(data_i8_func, "OS2-accu-15");
                if any(idx)
                    tflops_Oz2_i15(n == SIZE) = data_i8_tflops(idx);
                end

                idx = data_i8_m == n & data_i8_k == n & strcmp(data_i8_func, "OS2-fast-16");
                if any(idx)
                    tflops_Oz2_i16(n == SIZE) = data_i8_tflops(idx);
                end
            end

            if flag_f8
                idx = data_f8_m == n & data_f8_k == n & strcmp(data_f8_func, "OS2-accu-12");
                if any(idx)
                    tflops_Oz2_f12(n == SIZE) = data_f8_tflops(idx);
                end

                idx = data_f8_m == n & data_f8_k == n & strcmp(data_f8_func, "OS2-fast-13");
                if any(idx)
                    tflops_Oz2_f13(n == SIZE) = data_f8_tflops(idx);
                end
            end
        end

        nexttile;
        hold on;
        if flag_i8 || flag_f8
            x = max( [nnz(isfinite(tflops_64f(:))), nnz(isfinite(tflops_Oz1_7(:))), nnz(isfinite(tflops_Oz2_i15(:))), nnz(isfinite(tflops_Oz2_i16(:)))] );

            plot(1:length(SIZE), tflops_64f    , mark(1,2,1), 'DisplayName', "native FP64 " + upper(t)+"GEMM",        'MarkerSize', 5, 'LineWidth', 1);
            plot(1:length(SIZE), tflops_Oz1_7  , mark(1,1,2), 'DisplayName', "INT8-based Ozaki-I (7 slices)",         'MarkerSize', 5, 'LineWidth', 1);
            plot(1:length(SIZE), tflops_Oz2_i16, mark(1,3,3), 'DisplayName', "INT8-based Ozaki-II fast  (16 moduli)", 'MarkerSize', 5, 'LineWidth', 1);
            plot(1:length(SIZE), tflops_Oz2_i15, mark(1,4,4), 'DisplayName', "INT8-based Ozaki-II accu. (15 moduli)", 'MarkerSize', 5, 'LineWidth', 1);
            plot(1:length(SIZE), tflops_Oz2_f13, mark(1,5,5), 'DisplayName', "FP8-based Ozaki-II fast  (13 moduli)",  'MarkerSize', 5, 'LineWidth', 1);
            plot(1:length(SIZE), tflops_Oz2_f12, mark(1,7,6), 'DisplayName', "FP8-based Ozaki-II accu. (12 moduli)",  'MarkerSize', 5, 'LineWidth', 1);

            xlim([1 x]);
            xticks(1:x);
            xticklabels("2^{" + log2(SIZE) + "}");
            xtickangle(0)
            ylim('padded');
            yl_tmp = ylim;
            yl{ceil(find(gpu==GPUs)/2)}(1) = min(yl{ceil(find(gpu==GPUs)/2)}(1), yl_tmp(1));
            yl{ceil(find(gpu==GPUs)/2)}(2) = max(yl{ceil(find(gpu==GPUs)/2)}(2), yl_tmp(2));
        end

        grid on;
        set(gca,'FontName','Yu Gothic UI Semibold');
        title({GPUNAME(gpu==GPUs), SOFT(gpu==GPUs)});
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

    lgd = legend('Interpreter','tex','FontName','Yu Gothic UI Semibold','IconColumnWidth',15,'NumColumns',2);
    lgd.Layout.Tile = 'north';
    xlabel(tile,'\itm = n = k', 'Interpreter', 'tex');
    ylabel(tile,'TFLOP/s');
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
    tile.TileSpacing = "tight";
    tile.Padding = "compact";

    savefig(fig,"flops_multArch_"+t);
    exportgraphics(fig,"flops_multArch_"+t+".png",'Resolution',600);
end

end