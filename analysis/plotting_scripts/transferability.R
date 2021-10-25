library(ggplot2)
library(scales)
library(dict)


#fig_width <- 55
#fig_height <- 34
#basesize <- 70
#big_text_size <- 22
#small_text_size <- 16

basesize <- 45
big_text_size <- 11
small_text_size <- 8
fig_width <- 40
fig_height <- 10


text_color = "black"



transfer_plot <- function(dataset, epsilon, title, explainer) {
    input_file     <- sprintf("../results/analysis_transferability/%s/transfer_%s_eps=%s.csv", dataset, explainer, epsilon)

    save_path <- "./results/transferability"
    
    dir.create(save_path, showWarnings = FALSE)

    output_file <- sprintf("%s/%s_%s_eps=%s.pdf", save_path, dataset, explainer, epsilon)

    df  <- read.csv(input_file, header=T)

    ggplot(data = df, aes(x=student, y=master, fill=ifelse(check == "yes", fidelity, NA))) + 

    geom_tile() +

    geom_text(aes(label = ifelse(check == "yes", round(fidelity, 2), "")), size=big_text_size, color = text_color) +

    geom_text(aes(label = ifelse(check == "yes", round(unfairness, 2), "")), nudge_x=0.30, nudge_y=-0.30, size=small_text_size, color = text_color) +

    geom_text(aes(label = ifelse(check == "yes", round(label_agreement/100, 2), "")), nudge_x=0.30, nudge_y=0.30, size=small_text_size, color = text_color) +

    scale_fill_gradient(low = "white",high = "steelblue") +

    theme_minimal(base_size=basesize) + 

    theme(
        legend.position = 'right', 
        #legend.spacing.x = unit(0.5, 'cm'),
        #legend.text = element_text(margin = margin(t = 10))
        ) +
        
        guides(fill = guide_colorbar(
                               title = "Fidelity",
                               #label.position = "bottom",
                               title.position = "top", 
                               title.vjust = 1,
                               frame.colour = "black",
                               barwidth = 3.0,
                               barheight = 30.5
                               )) +


    labs(x = "Student model", y = "Teacher model", fill = "Fidelity", title = title) +

    facet_wrap(~metric, scales="free_x", ncol=4) + theme(panel.spacing = unit(1.2, "lines"))

    ggsave(output_file, dpi=300, width=fig_width, height=fig_height, limitsize = FALSE)
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

epsilons <- c(0.03, 0.05, 0.1)
#epsilons <- c(0.05)
datasets <- c("adult_income", "compas", "default_credit", "marketing")
explainers <- c("rl", "lm", "dt")

for (eps in epsilons){
    for (dataset in datasets){
        for (explainer in explainers){
            transfer_plot(dataset, eps, sprintf("%s -- %s -- eps=%s", dataset_map[[dataset]], explainer_map[[explainer]], eps), explainer)
        }
    }
}






