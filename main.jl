using Bio.Structure: coords, downloadpdb, PDB, read, resname, atoms, resnumber, serial, standardselector, AminoAcidSequence

function get_protein_names()
    ["1EN2"]
end

function download_proteins()
    protein_dataset = get_protein_names()
    for protein_name in protein_dataset
        if !isfile("protein_data/$protein_name.pdb")
            downloadpdb(protein_name, pdb_dir="protein_data")
        end
    end
end

function main()
    download_proteins()
    for protein_name in get_protein_names()
        protein = read("protein_data/$protein_name.pdb", PDB)
        for residue in protein["A"]
            name = resname(residue)
            println(name)
            for atom in atoms(residue)
                location = coords(atom[2])
                serial = serial(atom[2])
            end
        end
        seq = AminoAcidSequence(struct["A"], standardselector)
    end
end
