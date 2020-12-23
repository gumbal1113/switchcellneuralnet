% plot_tfposneg() makes a figure showing true positives, true negatives,
% false positives, and false negatives
function y = plot_tfposneg(figure_name,tp,fp,tn,fn,gstar,gkatp_mean_value)

if size(tp) > 0
    scatter(tp(1,:),tp(2,:),'bo'); hold on;
end
if size(tn) > 0
    scatter(tn(1,:),tn(2,:),'b*'); hold on;
end
if size(fp) > 0
    scatter(fp(1,:),fp(2,:),25,fp(3,:),'*'); hold on;
end
if size(fn) > 0
    scatter(fn(1,:),fn(2,:),25,fn(3,:),'o'); hold on;
end

colormap(jet);

gstar = gstar/gkatp_mean_value;

line([gstar gstar],[0 200],'LineWidth',1); hold on;
line([0 200],[gstar gstar],'LineWidth',1); hold on;

plot(gkatp_mean_value/gkatp_mean_value,gkatp_mean_value/gkatp_mean_value,'b.'); hold on;

title(figure_name);

xlabel('gkatp1');
ylabel('gkatp2');

if size(tp) > 0
    if size(fp) > 0
        if size(tn) > 0
            if size(fn) > 0
                legend({'true positive','true negative','false positive','false negative'});
            else
                legend({'true positive','true negative','false positive'});
            end
        else % no tn
            if size(fn) > 0
                legend({'true positive','false positive','false negative'});
            % no tn & no fn
            else
                legend({'true positive','false positive'});
            end
        end
    else % no fp
        if size(tn) > 0
            if size(fn) > 0
                legend({'true positive','true negative','false negative'});
            else % no fp && no fn
                legend({'true positive','true negative'});
            end
        else % no fp && no tn
            if size(fn) > 0
                legend({'true positive','false negative'});
            else % no fp && no tn && no fn
                legend({'true positive'});
            end
        end
    end
else % no tp
    if size(fp) > 0
        if size(tn) > 0
            if size(fn) > 0
                legend({'true negative','false positive','false negative'});
            else
                legend({'true negative','false positive'});
            end
        else % no tn
            if size(fn) > 0
                legend({'false positive','false negative'});
            % no tn & no fn
            else
                legend({'false positive'});
            end
        end
    else % no fp
        if size(tn) > 0
            if size(fn) > 0
                legend({'true negative','false negative'});
            else % no fp && no fn
                legend({'true negative'});
            end
        else % no fp && no tn
            if size(fn) > 0
                legend({'false negative'});
            end
        end
    end
end

xlim([0.6 1.55]);%[90/gkatp_mean_value 160/gkatp_mean_value]);
ylim([0.6 1.55]);%[90/gkatp_mean_value 160/gkatp_mean_value]);

colorbar;

hold off;