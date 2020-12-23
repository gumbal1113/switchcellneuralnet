function y = plot_expected_outcome(figure_name,s_islets,ns_islets,gstar,gkatp_mean_value)

if size(s_islets) > 0
    scatter(s_islets(1,:),s_islets(2,:),25,s_islets(3,:)*-50,'o'); hold on;
end
if size(ns_islets) > 0
    scatter(ns_islets(1,:),ns_islets(2,:),'*'); hold on;
end

colormap(jet);
colorbar;

gstar = gstar/gkatp_mean_value;

line([gstar gstar],[0 200],'LineWidth',1);
line([0 200],[gstar gstar],'LineWidth',1);

plot(gkatp_mean_value/gkatp_mean_value,gkatp_mean_value/gkatp_mean_value,'b.');

xlim([0.6 1.55]);%[90/gkatp_mean_value 160/gkatp_mean_value]);
ylim([0.6 1.55]);%[90/gkatp_mean_value 160/gkatp_mean_value]);

title(figure_name);

xlabel('gkatp1');
ylabel('gkatp2');

if size(s_islets) == 0
    legend({'non-switch'});
else
    if size(ns_islets) == 0
        legend({'switch'});
    else
        legend({'switch','non-switch'});
    end
end

hold off;

y = 1;
end