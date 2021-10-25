library(ggplot2)
library(scales)
library(dplyr)
library(dict)


basesize <- 60
big_text_size <- 20
small_text_size <- 18
fig_width <- 40
fig_height <- 32
text_color = "black"


half_unfairness <- function(group, explainer, title) {
    input_file     <- sprintf("../results/half_unfairness/%s_all_bbox_%s.csv", group, explainer)

    save_path <- "./results/half_unfairness"
    
    dir.create(save_path, showWarnings = FALSE)

    output_file <- sprintf("%s/%s_all_bbox_%s.png", save_path, group, explainer)


    df  <- read.csv(input_file, header=T)


    ggplot(data = df, aes(x=metric, y=dataset, fill=fidelity)) + 

    geom_tile() +

    geom_text(aes(label = round(fidelity, 2), color = text_color), size=big_text_size) +

    geom_text(aes(label = round(100*((fidelity - fidelity_uncons)/fidelity_uncons), 2), color = text_color), nudge_x=0.20, nudge_y=0.30, size=small_text_size) +

    scale_color_manual(guide = FALSE, values = c("black", "white")) +


    scale_fill_gradient(low = "white", high = "thistle") + 

    theme_minimal(base_size=basesize) + 

    theme(
        legend.position = 'right', 
        ) +
        
        guides(fill = guide_colorbar(
                               title = "Fidelity",
                               title.position = "top", 
                               frame.colour = "black",
                               barwidth = 3.0,
                               barheight = 20.5
                               )) +


    labs(x = "Metrics", y = "Datasets", title = title) +

    facet_wrap(~model_class, scales="free_x", ncol=2) + theme(panel.spacing = unit(1.2, "lines"))

    ggsave(output_file, dpi=300, width=fig_width, height=fig_height)
}


# dataset map
group_map <- dict()
group_map[["sg"]]   <- "Members"
group_map[["test"]] <- "Non-Members"


# explainer map
explainer_map <- dict()
explainer_map[["rl"]]   <- "Rule List"
explainer_map[["lm"]]   <- "Logistic Regression"
explainer_map[["dt"]]   <- "Decision Tree"

groups <- c("sg")
#explainers <- c("rl", "lm", "dt")
explainers <- c("lm")

for (group in groups){
    for (explainer in explainers){
        half_unfairness(group, explainer, sprintf("%s", explainer_map[[explainer]]))
    }
}

