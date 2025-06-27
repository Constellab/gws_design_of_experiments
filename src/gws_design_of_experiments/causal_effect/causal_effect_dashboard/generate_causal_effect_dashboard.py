from gws_core import (ConfigParams, Dashboard, DashboardType, OutputSpec,
                      OutputSpecs, StreamlitResource, Task, TaskInputs,
                      TaskOutputs, dashboard_decorator, task_decorator,
                      InputSpecs, ConfigSpecs, Folder, InputSpec)


@dashboard_decorator("CausalEffectDashboardDashboard", dashboard_type=DashboardType.STREAMLIT,
                     human_name="Generate show case app")
class CausalEffectDashboard(Dashboard):

    # retrieve the path of the app folder, relative to this file
    # the dashboard code folder starts with a underscore to avoid being loaded when the brick is loaded
    def get_app_folder_path(self):
        return self.get_app_folder_from_relative_path(__file__, "_causal_effect_dashboard")


@task_decorator("GenerateCausalEffectDashboard", human_name="Generate causal effect dashboard app",
                style=StreamlitResource.copy_style())
class GenerateCausalEffectDashboard(Task):
    """
    Task that generates the causal effect dashboard app.
    """

    input_specs = InputSpecs({'folder' : InputSpec(Folder)})
    output_specs = OutputSpecs({
        'streamlit_app': OutputSpec(StreamlitResource)
    })

    config_specs = ConfigSpecs({})

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        """ Run the task """

        streamlit_app = StreamlitResource()

        # set the input in the streamlit resource
        folder: Folder = inputs.get('folder')
        streamlit_app.add_resource(folder, create_new_resource=False)

        streamlit_app.set_dashboard(CausalEffectDashboard())
        streamlit_app.name = "Dashboard of Conditional Average Treatment Effect (CATE)"

        return {"streamlit_app": streamlit_app}
