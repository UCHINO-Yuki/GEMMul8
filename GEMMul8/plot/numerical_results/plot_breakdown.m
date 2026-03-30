function plot_breakdown(type_in,GPU_name)

arguments (Input)
    type_in (1,1) string = "d"
    GPU_name (1,1) string = "RTX5080"
end

FontSize = 8;

%% get data
dir_name_i8  = dir(GPU_name + "/oz2_results_i_" + type_in + "_time*");
m_i8 = [];
if ~isempty(dir_name_i8)
    file_name_i8 = GPU_name + "/" + dir_name_i8.name;
    data_i8 = detectImportOptions(file_name_i8);
    data_i8.SelectedVariableNames = 2;
    m_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 4;
    k_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 5;
    func_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 9;
    total_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 10;
    quant_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 11;
    gemms_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 12;
    requant_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 13;
    dequant_i8 = readmatrix(file_name_i8,data_i8);
    others_i8 = total_i8 - quant_i8 - gemms_i8 - requant_i8 - dequant_i8;
    quant_i8 = quant_i8./total_i8.*100;
    gemms_i8 = gemms_i8./total_i8.*100;
    requant_i8 = requant_i8./total_i8.*100;
    dequant_i8 = dequant_i8./total_i8.*100;
    others_i8 = others_i8./total_i8.*100;
end

m_f8 = [];
dir_name_f8  = dir(GPU_name + "/oz2_results_f_" + type_in + "_time*");
if ~isempty(dir_name_f8)
    file_name_f8 = GPU_name + "/" + dir_name_f8.name;
    data_f8 = detectImportOptions(file_name_f8);
    data_f8.SelectedVariableNames = 2;
    m_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 4;
    k_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 5;
    func_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 9;
    total_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 10;
    quant_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 11;
    gemms_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 12;
    requant_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 13;
    dequant_f8 = readmatrix(file_name_f8,data_f8);
    others_f8 = total_f8 - quant_f8 - gemms_f8 - requant_f8 - dequant_f8;
    quant_f8 = quant_f8./total_f8.*100;
    gemms_f8 = gemms_f8./total_f8.*100;
    requant_f8 = requant_f8./total_f8.*100;
    dequant_f8 = dequant_f8./total_f8.*100;
    others_f8 = others_f8./total_f8.*100;
end

%% plot
labels = ["quant" "gemms" "requant" "dequant" "others"];
size_list = [1024 2048 4096 8192 16384 32768];
for i=length(size_list):-1:1
    if size_list(i)>min(max(m_i8),max(m_f8))
        size_list(i)=[];
    end
end
xlims = unique(k_i8);
fig = figure('Position',[50,50,500,420]);
t = tiledlayout(4,length(size_list));

for tid = 1:length(size_list)
    m = size_list(tid);
    nexttile(tid); hold on; grid on;

    if ~isempty(dir_name_i8)

        idx = contains(func_i8,"fast-16") & m_i8 == m;
        if any(idx)

            colororder("glow");
            quant = quant_i8(idx,1);
            gemms = gemms_i8(idx,1);
            requant = requant_i8(idx,1);
            dequant = dequant_i8(idx,1);
            others = others_i8(idx,1);
            bar([quant,gemms,requant,dequant,others],'stacked')

        end

    end

    title("{\itm=n=2^{" + log2(m) + "}}");
    ylim([0 100]);
    yticks(0:20:100);
    if tid == 1
        ylabel({"INT8-fast","(16 moduli)"},'FontSize',FontSize);
        yticklabels(0:20:100);
    else
        yticklabels([]);
    end
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
    xticks(1:length(xlims));
    xticktxt = "2^{" + log2(xlims) + "}";
    xticktxt([2:floor(length(xticktxt)/2), floor(length(xticktxt)/2)+2:end-1])=""; 
    xticklabels(xticktxt);
    xtickangle(0)
    xlim([0.25 length(xlims)+0.75])
    ax = gca;
    ax.XAxis.FontSize = FontSize-2;
    ax.TickDir = 'out';

    if tid == length(size_list)
        yyaxis right
        ylim([0 100]);
        yticks(0:20:100);
        yticklabels([]);
        ax = gca;
        ax.YAxis(2).Color = 'k';
    end
end

for tid = 1:length(size_list)
    m = size_list(tid);
    nexttile(tid+length(size_list)); hold on; grid on;

    if ~isempty(dir_name_i8)

        idx = contains(func_i8,"accu-15") & m_i8 == m;
        if any(idx)

            colororder("glow");
            quant = quant_i8(idx,1);
            gemms = gemms_i8(idx,1);
            requant = requant_i8(idx,1);
            dequant = dequant_i8(idx,1);
            others = others_i8(idx,1);
            bar([quant,gemms,requant,dequant,others],'stacked')

        end

    end

    title("{\itm=n=2^{" + log2(m) + "}}");
    ylim([0 100]);
    yticks(0:20:100);
    if tid == 1
        ylabel({"INT8-accu.","(15 moduli)"},'FontSize',FontSize);
        yticklabels(0:20:100);
    else
        yticklabels([]);
    end
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
    xticks(1:length(xlims));
    xticktxt = "2^{" + log2(xlims) + "}";
    xticktxt([2:floor(length(xticktxt)/2), floor(length(xticktxt)/2)+2:end-1])=""; 
    xticklabels(xticktxt);
    xtickangle(0)
    xlim([0.25 length(xlims)+0.75])
    ax = gca;
    ax.XAxis.FontSize = FontSize-2;
    ax.TickDir = 'out';

    if tid == length(size_list)
        yyaxis right
        ylim([0 100]);
        yticks(0:20:100);
        yticklabels([]);
        ax = gca;
        ax.YAxis(2).Color = 'k';
    end
end

for tid = 1:length(size_list)
    m = size_list(tid);
    nexttile(tid+2*length(size_list)); hold on; grid on;

    if ~isempty(dir_name_f8)

        idx = contains(func_f8,"fast-13") & m_f8 == m;
        if any(idx)

            colororder("glow");
            quant = quant_f8(idx,1);
            gemms = gemms_f8(idx,1);
            requant = requant_f8(idx,1);
            dequant = dequant_f8(idx,1);
            others = others_f8(idx,1);
            bar([quant,gemms,requant,dequant,others],'stacked')

        end

    end

    title("{\itm=n=2^{" + log2(m) + "}}");
    ylim([0 100]);
    yticks(0:20:100);
    if tid == 1
        ylabel({"FP8-fast","(13 moduli)"},'FontSize',FontSize);
        yticklabels(0:20:100);
    else
        yticklabels([]);
    end
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
    xticks(1:length(xlims));
    xticktxt = "2^{" + log2(xlims) + "}";
    xticktxt([2:floor(length(xticktxt)/2), floor(length(xticktxt)/2)+2:end-1])=""; 
    xticklabels(xticktxt);
    xtickangle(0)
    xlim([0.25 length(xlims)+0.75])
    ax = gca;
    ax.XAxis.FontSize = FontSize-2;
    ax.TickDir = 'out';

    if tid == length(size_list)
        yyaxis right
        ylim([0 100]);
        yticks(0:20:100);
        yticklabels([]);
        ax = gca;
        ax.YAxis(2).Color = 'k';
    end
end

for tid = 1:length(size_list)
    m = size_list(tid);
    nexttile(tid+3*length(size_list)); hold on; grid on;

    if ~isempty(dir_name_f8)

        idx = contains(func_f8,"accu-12") & m_f8 == m;
        if any(idx)

            colororder("glow");
            quant = quant_f8(idx,1);
            gemms = gemms_f8(idx,1);
            requant = requant_f8(idx,1);
            dequant = dequant_f8(idx,1);
            others = others_f8(idx,1);
            bar([quant,gemms,requant,dequant,others],'stacked')

        end

    end

    title("{\itm=n=2^{" + log2(m) + "}}");
    ylim([0 100]);
    yticks(0:20:100);
    if tid == 1
        ylabel({"FP8-accu.","(12 moduli)"},'FontSize',FontSize);
        yticklabels(0:20:100);
    else
        yticklabels([]);
    end
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
    xticks(1:length(xlims));
    xticktxt = "2^{" + log2(xlims) + "}";
    xticktxt([2:floor(length(xticktxt)/2), floor(length(xticktxt)/2)+2:end-1])=""; 
    xticklabels(xticktxt);
    xtickangle(0)
    xlim([0.25 length(xlims)+0.75])
    ax = gca;
    ax.XAxis.FontSize = FontSize-2;
    ax.TickDir = 'out';

    if tid == length(size_list)
        yyaxis right
        ylim([0 100]);
        yticks(0:20:100);
        yticklabels([]);
        ax = gca;
        ax.YAxis(2).Color = 'k';
    end
end

nexttile(1);
lgd = legend(labels,'FontSize',FontSize,'Interpreter','tex','FontName','Yu Gothic UI Semibold','IconColumnWidth',12,'NumColumns',5);
lgd.Layout.Tile = 'north';
t.TileSpacing = "tight";
t.Padding = "compact";
xlabel(t,"\itk");
ylabel(t,"%");
set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
ax = gca;
ax.XAxis.FontSize = FontSize-2;

savefig(fig,GPU_name+"/"+GPU_name+"_breakdown_"+type_in);
exportgraphics(fig,GPU_name+"/"+GPU_name+"_breakdown_"+type_in+".png",'Resolution',600);

end
