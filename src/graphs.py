import pandas as pd
import os
import matplotlib.pyplot as pl

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


'''
Takes in an int of number of graphs wanted 
made in order of data stored in out directory
figure consists of predicted presence and true presence
'''
def file_graph_temporal(num_graphs):

    filenames = os.listdir(os.path.join("data","out","separate_evaluations"))

    #region
    # print(filenames)
    # print(len(filenames))
    # print(filenames[0][:-4])
    # return
    #endregion

    os.makedirs(os.path.join("data","out","temporal_plots"))
    for i in range(num_graphs): #len(filenames)

        evals = pd.read_csv(os.path.join("data/out/separate_evaluations",filenames[i])) 

        title = evals['file'].unique()[0]

        to_plot = evals[['pred','label','temporal_frame_start_times']].copy(True) #,'cfnmtx'

        dfm = to_plot.melt("temporal_frame_start_times", var_name='bin', value_name='vals')

        viz = to_plot.copy(True)

        viz = viz.replace({'label':{0:0,1:1}})
        viz = viz.replace({'pred':{0:0,1:1}})

        dfm_viz = viz.melt("temporal_frame_start_times", var_name='bin', value_name='Presence')

        # comment out if absence is wanted on graph
        dfm_viz.drop(dfm_viz[dfm_viz['Presence'] == 0].index, inplace = True)

        # more chart info
        '''
        # dfm_viz.loc[(dfm_viz.bin == 'pred' ) & (dfm_viz.Presence == 0), "bin"] = "pred_absence"
        # dfm_viz.loc[(dfm_viz.bin == 'label' ) & (dfm_viz.Presence == 0), "bin"] = "true_absence"

        # dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'TP'), "bin"] = "TP"
        # dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'TN'), "bin"] = "TN"
        # dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'FP'), "bin"] = "FP"
        # dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'FN'), "bin"] = "FN"
        '''

        
        
        '''
        # g = sns.scatterplot(data=dfm_viz, x="temporal_frame_start_times", y="Presence", hue="bin")
        # g.set(rc={'figure.figsize':(12,8.27)})
        # g.set(xticklabels=[])  
        # g.set(yticklabels=[]) 
        # g.yaxis.set_major_locator(ticker.MultipleLocator(1)) 
        # g.set(title='Penguins: Body Mass by Species for Gender')
        '''
        

        #region
        # g.tick_params(bottom=False)
        # pl.show()
        #endregion
        # sns.set_theme()
        # g = sns.catplot(x="temporal_frame_start_times", y='bin', kind="swarm",height = 3,aspect = 6,data= dfm_viz)
        # g.fig.subplots_adjust(top=0.9)
        # g.fig.suptitle(title)
        # g.set(ylabel=None)
        # g.savefig(os.path.join("data/out/temporal_plots",filenames[i][:-4]+'_temporal_plot'))
        # plt.close('all')

        fig = px.scatter(dfm_viz, y='bin', x="temporal_frame_start_times", color="bin", width=5*800,height=400,)
        fig.update_traces(marker_size=5)
        fig.update_layout(title_text = title, title_x=0.5,legend_title_text='Annotation',yaxis_title=None)
        # fig.update_yaxes(visible=False, showticklabels=True)
        # fig.show()
        fig.write_image(file=os.path.join("data/out/temporal_plots",filenames[i][:-4]+'_temporal_plot.png'), format='png')
        fig.write_html(file=os.path.join("data/out/temporal_plots",filenames[i][:-4]+'_temporal_plot.html'))

# file_graph_temporal(2)

'''
Takes in an int of the number of graphs wanted
also made in order of data stored in out directory
figure consists of predicted presence and true presence
as well as the Rate of the data (T/F,P/N) per data predicted
on a scale of .02 frames per prediction.
'''
def file_graph_temporal_rates(num_graphs):

    filenames = os.listdir(os.path.join("data","out","separate_evaluations"))

    #region
    # print(filenames)
    # print(filenames[0][:-4])
    # os.makedirs(os.path.join("data","out","temporal_plots"))
    #endregion

    for i in range(num_graphs): #len(filenames)
        evals = pd.read_csv(os.path.join("data/out/separate_evaluations",filenames[i])) 

        title = evals['file'].unique()[0]

        to_plot = evals[['pred','label','temporal_frame_start_times','cfnmtx']].copy(True) #,'cfnmtx'

        dfm = to_plot.melt("temporal_frame_start_times", var_name='bin', value_name='vals')

        viz = to_plot.copy(True)

        viz = viz.replace({'label':{0:0,1:1}})
        viz = viz.replace({'pred':{0:0,1:1}})

        dfm_viz = viz.melt("temporal_frame_start_times", var_name='bin', value_name='Presence')

        # comment out if absence is wanted on graph
        dfm_viz.drop(dfm_viz[dfm_viz['Presence'] == 0].index, inplace = True)
        # dfm_viz.drop(dfm_viz[dfm_viz['Presence'] == 1].index, inplace = True)

        # more chart info
        '''
        # dfm_viz.loc[(dfm_viz.bin == 'pred' ) & (dfm_viz.Presence == 0), "bin"] = "pred_absence"
        # dfm_viz.loc[(dfm_viz.bin == 'label' ) & (dfm_viz.Presence == 0), "bin"] = "true_absence"
        '''

        dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'TP'), "bin"] = "TP"
        dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'TN'), "bin"] = "TN"
        dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'FP'), "bin"] = "FP"
        dfm_viz.loc[(dfm_viz.bin == 'cfnmtx' ) & (dfm_viz.Presence == 'FN'), "bin"] = "FN"

    
        '''
        # g = sns.scatterplot(data=dfm_viz, x="temporal_frame_start_times", y="Presence", hue="bin")
        # g.set(rc={'figure.figsize':(12,8.27)})
        # g.set(xticklabels=[])  
        # g.set(yticklabels=[]) 
        # g.yaxis.set_major_locator(ticker.MultipleLocator(1)) 
        # g.set(title='Penguins: Body Mass by Species for Gender')
        '''
        
        # g.tick_params(bottom=False)
        # pl.show()
        # sns.set_theme()
        # g = sns.catplot(x="temporal_frame_start_times", y='bin', kind="swarm",height = 3,aspect = 6,data= dfm_viz)
        # g.fig.subplots_adjust(top=0.9)
        # g.fig.suptitle(title)
        # g.set(ylabel=None)
        # g.savefig(os.path.join("data/out/temporal_plots",filenames[i][:-4]+'_temporal_plot_rate'))
        # plt.close('all')

        fig = px.scatter(dfm_viz, y='bin', x="temporal_frame_start_times", color="bin", width=5*800,height=400,)
        fig.update_traces(marker_size=5)
        fig.update_layout(title_text = title, title_x=0.5,legend_title_text='Annotation',yaxis_title=None)
        # fig.update_yaxes(visible=False, showticklabels=True)
        # fig.show()
        fig.write_image(file=os.path.join("data/out/temporal_plots",filenames[i][:-4]+'_temporal_plot_rates.png'), format='png')
        fig.write_html(file=os.path.join("data/out/temporal_plots",filenames[i][:-4]+'_temporal_plot_rates.html'))
