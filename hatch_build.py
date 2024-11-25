from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface

import tomlkit
import pathlib


class PinVersionMetadataHook(MetadataHookInterface):
    """Update the dependency versions so they are pinned metadata of the wheel artefact produced by a wheel."""
    def update(self, metadata):
        change_count = 0
        # Overlay the pinned dependencies, replacing dependencies if they already exist.
        for pinned_dep in self.config["config"]["pinned_dependencies"]:
            pinned_dep_name = pinned_dep.split(" ")[0]
            for dep in metadata["dependencies"]:
                dep_name = dep.split(" ")[0]
                # If the dependency is already in the list, replace it with the pinned version.
                if dep_name == pinned_dep_name:
                    index = metadata["dependencies"].index(dep)
                    metadata["dependencies"][index] = pinned_dep
                    change_count += 1
                    break
        print(f"Updated {change_count} dependencies to pinned versions in hatch's metadata "
              f"for building wheel artefact metadata.")


class PinMetadataBuildHook(BuildHookInterface):
    """Update the dependency versions so they are pinned in the pyproject.toml of the sdist artefact produced by a build."""
    PLUGIN_NAME = "pin-during-build"

    original_dependencies: list = None
    made_changes: bool = False

    def initialize(self, version, build_data):
        # Get the pinned dependencies, this is a list of package specifier strings that are the dependencies to pin.
        pinned_dependencies = self.metadata.hatch.metadata.hook_config["custom"]["config"]["pinned_dependencies"]

        # Load the toml file
        pyproject_file = pathlib.Path("pyproject.toml")
        toml_data = tomlkit.loads(pyproject_file.read_text())

        # Get the dependencies from the toml file, this is a list of package specifier strings.
        dependencies = toml_data["project"]["dependencies"]

        # Save the original dependencies for later.
        self.original_dependencies = dependencies.copy()

        change_count = 0

        # Overlay the pinned dependencies, replacing dependencies if they already exist.
        for pinned_dep in pinned_dependencies:
            pinned_dep_name = pinned_dep.split(" ")[0]
            for dep in dependencies:
                dep_name = dep.split(" ")[0]
                # If the dependency is already in the list, replace it with the pinned version.
                if dep_name == pinned_dep_name:
                    index = dependencies.index(dep)
                    dependencies[index] = pinned_dep
                    change_count += 1
                    break

        # If we made any changes, write the changes back to the pyroject.toml file for the build.
        if change_count > 0:
            # Write the changes back to the file.
            pyproject_file.write_text(tomlkit.dumps(toml_data))
            print(f"Updated {change_count} dependencies to pinned versions in the pyproject.toml "
                  f"for building sdist artefact.")
            # Set a flag to restore the original dependencies after the build.
            self.made_changes = True

        # Update the build data with the pinned dependencies so it populates the METADATA file in the wheel artefact.
        build_data["dependencies"] = dependencies

        return super().initialize(version, build_data)

    def finalize(self, version, build_data, artefact_path):
        # If we have made changes restore the original dependencies after the build to keep the git repo clean.
        if self.made_changes:
            pyproject_file = pathlib.Path("pyproject.toml")
            toml_data = tomlkit.loads(pyproject_file.read_text())
            toml_data["project"]["dependencies"] = self.original_dependencies
            pyproject_file.write_text(tomlkit.dumps(toml_data))
            print("Restored original dependencies in pyproject.toml file after building.")
