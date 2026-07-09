function plot_breakdown(gpu)

arguments (Input)
    gpu (1,1) string = "RTX5080"
end

FontSize = 8;
type_in = "d";

%% get data
dir_name = dir(gpu + "/oz2_results_" + type_in + "gemm_time*");
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

idx = find(contains(data.VariableNames,"total"));
data.SelectedVariableNames = idx;
data_total = readmatrix(file_name,data);

idx = find(contains(data.VariableNames,"scaling") & ~contains(data.VariableNames,"undo"));
data.SelectedVariableNames = idx;
data_quant = readmatrix(file_name,data);

idx = find(contains(data.VariableNames,"low_prec"));
data.SelectedVariableNames = idx;
data_gemms = readmatrix(file_name,data);

idx = find(contains(data.VariableNames,"mod"));
data.SelectedVariableNames = idx;
data_requant = readmatrix(file_name,data);

idx = find(contains(data.VariableNames,"undo"));
data.SelectedVariableNames = idx;
data_dequant = readmatrix(file_name,data);

data_others = data_total ...
    - data_quant ...
    - data_gemms ...
    - data_requant ...
    - data_dequant;

data_quant = data_quant./data_total.*100;
data_gemms = data_gemms./data_total.*100;
data_requant = data_requant./data_total.*100;
data_dequant = data_dequant./data_total.*100;
data_others = data_others./data_total.*100;

if contains(gpu,"RTX")
    idx = data_m < 16384;
    data_m = data_m(idx);
    data_k = data_k(idx);
    data_func = data_func(idx);
    data_quant = data_quant(idx);
    data_gemms = data_gemms(idx);
    data_requant = data_requant(idx);
    data_dequant = data_dequant(idx);
    data_others = data_others(idx);
end

%% plot
labels = ["quant" "gemms" "requant" "dequant" "others"];
size_list = [1024 2048 4096 8192 16384 32768];
for i=length(size_list):-1:1
    if size_list(i)>max(data_m(contains(data_func,"OS2-i8-fast-15")))
        size_list(i)=[];
    end
end
xlims = unique(data_k);
fig = figure('Position',[50,50,500,370]);
t = tiledlayout(4,length(size_list));

for tid = 1:length(size_list)
    m = size_list(tid);
    nexttile(tid); hold on; grid on;

    idx = contains(data_func,"OS2-i8-fast-15") & data_m == m;
    if any(idx)

        colororder("glow");
        quant = data_quant(idx,1);
        gemms = data_gemms(idx,1);
        requant = data_requant(idx,1);
        dequant = data_dequant(idx,1);
        others = data_others(idx,1);
        bar([quant,gemms,requant,dequant,others],'stacked')

    end

    title("{\itm=n=2^{" + log2(m) + "}}");
    ylim([0 100]);
    yticks(0:20:100);
    if tid == 1
        ylabel({"INT8-fast","(15 moduli)"},'FontSize',FontSize);
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

    idx = contains(data_func,"OS2-i8-accu-15") & data_m == m;
    if any(idx)

        colororder("glow");
        quant = data_quant(idx,1);
        gemms = data_gemms(idx,1);
        requant = data_requant(idx,1);
        dequant = data_dequant(idx,1);
        others = data_others(idx,1);
        bar([quant,gemms,requant,dequant,others],'stacked')

    end

    % title("{\itm=n=2^{" + log2(m) + "}}");
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

    idx = contains(data_func,"OS2-f8-fast-12") & data_m == m;
    if any(idx)

        colororder("glow");
        quant = data_quant(idx,1);
        gemms = data_gemms(idx,1);
        requant = data_requant(idx,1);
        dequant = data_dequant(idx,1);
        others = data_others(idx,1);
        bar([quant,gemms,requant,dequant,others],'stacked')

    end

    % title("{\itm=n=2^{" + log2(m) + "}}");
    ylim([0 100]);
    yticks(0:20:100);
    if tid == 1
        ylabel({"FP8-fast","(12 moduli)"},'FontSize',FontSize);
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

    idx = contains(data_func,"OS2-f8-accu-12") & data_m == m;
    if any(idx)

        colororder("glow");
        quant = data_quant(idx,1);
        gemms = data_gemms(idx,1);
        requant = data_requant(idx,1);
        dequant = data_dequant(idx,1);
        others = data_others(idx,1);
        bar([quant,gemms,requant,dequant,others],'stacked')

    end

    % title("{\itm=n=2^{" + log2(m) + "}}");
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
xlabel(t,"\itk", 'Interpreter', 'tex','FontName','Yu Gothic UI Semibold');
ylabel(t,"%",'FontName','Yu Gothic UI Semibold');
set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
ax = gca;
ax.XAxis.FontSize = FontSize-2;

savefig(fig,gpu+"/"+gpu+"_breakdown_"+type_in);
exportgraphics(fig,gpu+"/"+gpu+"_breakdown_"+type_in+".png",'Resolution',600);

end
