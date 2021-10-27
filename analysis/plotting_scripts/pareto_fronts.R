library(ggplot2)
library(scales)
library(ggpubr)
library(dict)
source("theme.R")

point_size <- 5.5
line_size <- 0.5
point_size_vline <- 1.5

alpha_line <- 0.99
alpha_point <- 1.0
alpha_ribbon <- 0.15


fig_width       <- 48
fig_height      <- 24

basesize <- 70 


pareto_plot <- function(dataset, title, explainer) {
    
    input_file              <- sprintf("../results/pareto_merged/%s/%s_%s.csv", dataset, dataset, explainer)
    input_file_baseline     <- sprintf("../results/unfairness_bbox/%s.csv", dataset)

    save_path <- "./graphs/pareto"
    
    dir.create(save_path, showWarnings = FALSE, recursive = TRUE)

    output_file <- sprintf("%s/%s_%s.pdf", save_path, dataset, explainer)

    df                      <- read.csv(input_file, header=T)
    df_baseline             <- read.csv(input_file_baseline, header=T)

    pp <- ggplot() + 

    geom_line(data=df, aes(x=unfairness, y=fidelity, color=model_class),  size=line_size, linetype = "dashed", alpha=alpha_line) + 

    geom_point(data=df, aes(x=unfairness, y=fidelity, color=model_class, shape=model_class), size=point_size, alpha=alpha_point) + 

    geom_vline(data = df_baseline, mapping = aes(xintercept = unfairness, color=model_class), size=point_size_vline) +

    geom_ribbon(data=df, aes(x=unfairness, ymin= fidelity - fidelity_std, ymax= fidelity + fidelity_std, fill=model_class), linetype = "dashed", alpha=alpha_ribbon) +

    theme_light_v2(base_size=basesize) + 

    scale_color_manual(
        name="Model",
        values=c("#0e5357", "#cda815", "#a83546", "#a13fb6")
    )  +

    scale_fill_manual(
        name="Model",
        values=c("#0e5357", "#cda815", "#a83546", "#a13fb6")
    ) +

    scale_shape_manual(
        name="Model",
        values=c(20, 20, 20, 20)
    )  +
    
    labs(x = "Unfairness", y = "Fidelity", title = title) + 


    facet_grid(group~metric, scales = "free_x") + theme(panel.spacing.x = unit(8.0, "lines"), panel.spacing.y = unit(8.0, "lines")) +

    scale_x_continuous(

        breaks = pretty_breaks(n = 3)
    
    ) + 

    theme(
        legend.direction = "horizontal", 
        legend.position = "top", 
        legend.box = "horizontal")  +

    
    labs(color='')


    pp 


    ggsave(output_file, dpi=300, width=fig_width, height=fig_height)
                    
}



# dataset map
dataset_map <- dict()
dataset_map[["adult_income"]]   <- "Adult Income"
dataset_map[["compas"]]         <- "COMPAS"
dataset_map[["default_credit"]] <- "Default Credit"
dataset_map[["marketing"]]      <- "Marketing"

# explainer map
explainer_map <- dict()
explainer_map[["rl"]]   <- "Rule List"
explainer_map[["lm"]]   <- "Logistic Regression"
explainer_map[["dt"]]   <- "Decision Tree"

datasets <- c("adult_income", "compas", "default_credit", "marketing")
explainers <- c("rl", "lm", "dt")
for (dataset in datasets){
    for (explainer in explainers){
            pareto_plot(dataset, sprintf("%s -- %s", dataset_map[[dataset]], explainer_map[[explainer]]), explainer)
    }
}
