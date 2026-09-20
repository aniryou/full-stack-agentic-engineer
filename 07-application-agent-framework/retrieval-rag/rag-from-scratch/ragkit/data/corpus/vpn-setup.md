# VPN Setup Guide

## Client
The approved VPN client is Meridian Connect. Install it from the internal app
catalogue; personal or third-party VPN clients are not permitted for company
systems.

## Supported Systems
Meridian Connect supports macOS 13 or later, Windows 11, and Ubuntu 22.04 or
later. Split tunneling is disabled by policy, so all traffic is routed through
the VPN while connected. Connect to the gateway at vpn.meridian.internal.

## Troubleshooting
If you see error ERR_4290 when connecting, your device certificate has expired.
Re-enrol the device in the app catalogue to issue a fresh certificate, then
reconnect. Persistent failures should be raised with the IT service desk.
