using CSV
using DataFrames
using JLD2

const NUMBER_FORMAT = AbstractString

function DataRow(batch_number::Integer, number_format::NUMBER_FORMAT, epoch_count::Integer, epochs_accuracy::AbstractArray, set_sampled::String)
    return (Training_Batch = batch_number,
           Number_Format = number_format,
           Set_Sampled = set_sampled,
           [Symbol(i - 1 == 0 ? "Pre-Training" : "Epoch_$(i-1)") =>
           (epochs_accuracy[i] == 0.0 ? missing : epochs_accuracy[i]) for i in 1:(epoch_count+1)]...) 
end

function ParseControllerAsRows(path::String, batch_number::Integer, number_format::NUMBER_FORMAT)
    results = JLD2.load(path)
    num_epochs = length(results["epochs_accuracy"][1]) - 1
    return (DataRow(batch_number, number_format, num_epochs, results["epochs_accuracy"][1], "training"),
            DataRow(batch_number, number_format, num_epochs, results["epochs_accuracy"][2], "testing"))
end

function CSVPrinter(report_path::String, typeset::Vector)
    logging = ""
    try
        new_batch = CSV.read(report_path * "testing_logs.csv", DataFrame)[end, 1] + 1
        CSV.write(report_path * "testing_logs.csv",
                 [ParseControllerAsRows(report_path * "$i/progress_controller.jld2", new_batch, i)[j]
                  for i in typeset for j in 1:2]; append = true)
    catch e
        logging = "Alert: $e"
        println("Alert: $e")
        try
            CSV.write(report_path * "testing_logs.csv",
                    [ParseControllerAsRows(report_path * "$i/progress_controller.jld2", 1, i)[j]
                    for i in typeset for j in 1:2])
        catch e2
            logging = logging * "\n SubAlert: $e2"
        end
    end
    # Once we're done with the files, delete 'em
    for type in typeset
        try
            rm(report_path * "$type/progress_controller.jld2")
        catch
            logging = logging * "\nCould not delete file: " * report_path * "$type/progress_controller.jld2." *
                                "Verify that the log existed to read."
        end
    end
    return logging
end