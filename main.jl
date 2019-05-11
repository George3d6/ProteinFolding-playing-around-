using Bio.Structure

function main()
    protein_dataset = ["1EN2"]
    for protein_name in protein_dataset
        if !isfile("protein_data/$protein_name.pdb")
            downloadpdb(protein_name)
        end
    end
end
