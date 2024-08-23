using PyCall;
using PyPlot;
PyDict(matplotlib["rcParams"])["font.size"] = 22
PyDict(matplotlib["rcParams"])["pdf.fonttype"] = Int64(42)
PyDict(matplotlib["rcParams"])["axes.spines.right"] = false
PyDict(matplotlib["rcParams"])["axes.spines.top"] = false
PyDict(matplotlib["rcParams"])["axes.linewidth"] = 2
PyDict(matplotlib["rcParams"])["figure.figsize"] = (5, 4)

sns = pyimport("seaborn");

fluo_colors_nacc = ["#012a4a", "#014f86", "#468faf", "#a9d6e5"]
fluo_colors_dms = ["#621b00", "#bc3908", "#ff9e00", "#ffcd7d"]
fluo_colors_dls = ["#1e441e", "#2a7221", "#31cb00", "#96e072"]

function check_pval(pval)
    if pval < 1e-12
        pstr = "888"
    elseif pval < 1e-11
        pstr = "< 1e-11"
    elseif pval < 1e-10
        pstr = "< 1e-10"
    elseif pval < 1e-9
        pstr = "< 1e-9"
    elseif pval < 1e-8
        pstr = "< 1e-8"
    elseif pval < 1e-7
        pstr = "< 1e-7"
    elseif pval < 1e-6
        pstr = "< 1e-6"
    elseif pval < 0.00001
        pstr = "< 1e-5"
    elseif pval < 0.0001
        pstr = "< 1e-4"
    elseif pval < 0.001
        pstr = "< 1e-3"
    else
        pstr = string(pval)
    end
    return pstr
end

function pval_stars(pval)
    if pval < 0.0001
        pstr = "***"
    elseif pval < 0.001
        pstr = "**"
    elseif pval < 0.05
        pstr = "*"
    else 
        pstr = ""
    end
    return pstr
end