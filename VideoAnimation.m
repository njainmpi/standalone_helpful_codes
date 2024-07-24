%% Display Line Animation
% Create the initial animated line object. Then, use a loop to add 1,000 points 
% to the line. After adding each new point, use |drawnow| to display the new point 
% on the screen.
clf
h = animatedline('Color','#0072BD','LineWidth',4);
% g = animatedline('Color','#D95319','LineWidth',4);
f = animatedline('Color','#D95319','LineWidth',8);
axis([0,75,-0.6,0.55])

y=pial_lowres_0164ms;
% z=vc_highres_0164ms;
x=Range;
xlabel 'Time (in sec)'
ylabel 'Percent Signal Change'
legend ('Pial Surface', 'Visual Cortex')
fontsize(30,"points")
set(gcf,'color','w')

for k = 1:27
    % pause(0.05)
    addpoints(h,x(k),y (k)); 
    % hold on
    % addpoints(g,x(k),z(k));
    % 
    for ii=7:17
        addpoints (f, x(ii), -0.5)
    end
    drawnow

    exportgraphics(gcf,'PSC_vaso.gif','Append',true);
      
end


%% 
% For faster rendering, add more than one point to the line each time through 
% the loop or use |drawnow limitrate|.
% 
% Query the points of the line.
%%
% [xdata,ydata] = getpoints(h);
% %% 
% % Clear the points from the line.
% %%
% % clearpoints(h)
% drawnow
%% 
% Copyright 2015-2017 The MathWorks, Inc.
