Types = {'Full' 'Fixed'};
Distributions = {'Uniform','Gaussian','Line','GreenGenes'};

for Type = Types
    Type = Type{1};
    useNamedFigure('Performance'); clf; hold on;
    for Name = Distributions
        Name= Name{1};
        Data = load(['../Output/Performance/' Name '-' Type '.dat']);
        plot(Data(:,1),Data(:,2));
    end
    
    legend(Distributions{:},'Location','northwest');
    
    xlabel('Problem Size');
    ylabel('Run Time (sec)');
    title(sprintf('Run Time For %s Barcodes',Type));

    prettyPlot;
    print('-dpng',[Type '-Performance.png']);
end

